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

### `src/viewer` (three.js WebGL2 viewer)

* **API superset of §4.9** (documented signatures kept). Extras: `ViewerOptions`
  (`maxPixelRatio` 2, `antialias`, `interactivePointBudget` 1.5 M, `pickMode`
  `'auto'|'cpu'`, `autoFit` `'first'|'follow'|'off'`, `fovDeg`, `background: 'transparent'`);
  `setPointCloud(pc, { pointSizePx, splats, softShading, minConfidence })`;
  `setDisplay({ …, unlit, edlStrength, edlRadiusPx, sparsePointSizePx })`;
  `fitToContent(animate?)`; `pickPoint`, `pickPoints`, `measureBetweenScreenPoints(p1, p2)`,
  `measureTap(p)` (screen points = CSS px relative to the container, `{x,y}` or `[x,y]`);
  `worldToScreen`, `setOrbitTarget`/`getOrbitTarget` (world frame), `getWorldUp`,
  `getContentBounds`, `requestRender`, `renderNow`, `readPixels`, `getStats`, `framePending`.
* **Frames**: content lives under world roots rotated so world up → three +Y ('z':
  −90° about X, `(x,y,z) → (x,z,−y)`). `setWorldUp('auto')`: up from camera poses —
  smallest eigenvector of Σ xxᵀ of the camera x axes (phones are held level; IRLS
  Cauchy weights drop rolled frames), sign from the mean of the camera up vectors
  (−Rᵀe_y); falls back to that mean when the x axes are parallel / not level. Unlike
  the plain mean it is not biased by camera pitch on partial orbits. When up changes,
  the camera is carried along (view kept, horizon levelled). Grid only in 'z' mode.
* **Colour**: point / vertex colours are sRGB Uint8 and are converted to linear in the
  shaders (three treats vertex colours as linear); output via three's colour-space
  conversion, so direct and EDL paths match (exact sRGB value for unlit content).
* **EDL** (points only): content → offscreen SRGB8_ALPHA8 + depth texture
  (DEPTH_COMPONENT32F, fallback 24; no float colour target needed; if neither
  framebuffer is complete EDL is disabled and picking uses the CPU path). Composite:
  log2 of linearised depth, 8 neighbours at 1.4 CSS px, `exp(−300·s·mean(max(0, l₀−lᵢ)))`,
  writes `gl_FragDepth` so overlays depth-test against content. Mesh fragments write
  alpha 0 into the target (shared `uEdlWeight` uniform) and are not shaded; a mesh
  without points is drawn directly (canvas MSAA). Depth precision: near/far are
  recomputed every frame from a clip sphere (outlier-clamped full bounds ∪ cameras ∪
  grid), far/near ≤ ~2000 unless zoomed very close to the orbit target.
* **Points**: round splats; size = 2·radius·projScale/depth when `radii` exist and
  `splats` (default), else `pointSizePx`·dpr; clamped to [1, 64·dpr] device px (and
  the GL max). Dense arrays are wrapped, not copied (kept for context restore and CPU
  picking). Clouds > 1.2·budget get an extra index buffer (golden-ratio stride
  permutation, `budget` entries) drawn while the view moves (size × min(2, √(n/budget))).
* **Picking** (GPU, default; O(1) in the number of points, occlusion-correct):
  (A) content depth (normal splats) read in a ±14 CSS px window → front surface
  depth at the tapped pixel (or nearest covered pixel = tap in a gap); (B) a ×4
  magnified 64² render around the tap with 1 px splats → sub-pixel depths of point
  centres / mesh; samples behind pass-A depth ×1.05 are rejected; the hit is on the
  tap ray at the depth of a planar fit 1/z = a + b·u + c·v (exact for planes). Error on
  a 240×180 px test canvas viewed at 30° ≈ 5 mm at 7 m (≈ 0.1 px). A one-time
  known-pattern readback self-test guards against Safari readback noise (Private
  Browsing fingerprinting protection); on failure (or `pickMode: 'cpu'`, or no EDL
  target) the CPU picker projects all points (≈ 10–30 ms per 2 M points on one core),
  emulates splat coverage and uses the same planar fit; mesh vertices act as 1 px points.
* **Render loop**: rAF only on change (content / display / resize / controls 'change'
  incl. damping / target animation); `resize()` (ResizeObserver) redraws synchronously.
  Context loss: frames stop, three restores resources from the retained arrays.
  `dispose()` cancels the pending frame, removes listeners/DOM and forces context loss.

### `src/dense` — PatchMatch MVS (types, synthetic data, CPU reference, WebGPU)

* **Files**: `types.ts` (DenseView, DepthMap, DepthMapStore, MemoryDepthMapStore, `GeometricInput`,
  `PatchMatchOptions` + `DEFAULT_PATCHMATCH_OPTIONS`, `MAX_PATCHMATCH_SOURCES` = 16),
  `patchmatch-common.ts` (shared CPU/GPU set-up), `patchmatch-cpu.ts`, `patchmatch-wgsl.ts`,
  `patchmatch-gpu.ts`, `synthetic.ts`, `index.ts` (synthetic and low-level helpers are re-exported
  as the namespaces `synthetic` / `patchMatchInternals` to avoid name clashes in barrels).
* **Hypothesis / homography convention**: per pixel (z, n) in the reference camera frame, n facing the
  camera. Plane nᵀX + δ = 0 with δ = −z·n·K_r⁻¹p̃ > 0 (distance to the plane). With
  R_sr = R_s R_rᵀ, t_sr = t_s − R_sr t_r: **H = K_s (R_sr − t_sr nᵀ/δ) K_r⁻¹** (the §4.6 formula with
  d = δ; written as nᵀX = d the sign flips). Propagated planes are re-intersected with the pixel ray.
* **Cost**: 1 − bilateral-weighted NCC on Rec.709 grey (rounded to 8 bit, identical to the r8unorm
  upload); window offsets −r…r step s (ACMM; defaults 6×6 = 36 samples, centre excluded); weights
  exp(−|Δxy|/(2σs²) − ΔC/(2σc²)), σs = 5 px, σc = 3, where **ΔC = mean |ΔR|,|ΔG|,|ΔB|** (colour instead
  of ACMM's grey, to separate surfaces of equal luminance). Weighted variance < (0.75/255)² ⇒
  textureless ⇒ cost 2 and the pixel is not updated. Source samples are hardware-bilinear
  (clamp-to-edge); window centre outside the source / behind it ⇒ 2.
* **Propagation & view selection**: red/black passes, ACMH adaptive checkerboard (4 V-shaped regions of
  7 px, 4 strips of 11 px reaching 23 px). ACMH multi-hypothesis view selection over **9** hypotheses
  (8 region winners + the current one): good if c < 0.8·exp(−t²/90), bad if c > 1.2; view selected if
  #good > 2 and #bad < 3, weight = mean exp(−c²/0.18) of its good costs × soft prior
  (0.3 + 0.7·fraction of the 4-neighbours that selected it). Fallback when nothing is selected: the
  `topK` (3) views with the lowest best cost, weight 1. Refinement: 6 proposals (perturbed / random
  depth × normal; relative depth perturbation 5 %·0.5ᵗ, floor 0.2 %; normal 0.15 rad·0.5ᵗ); random
  depths are uniform in **inverse depth** (init and proposals).
* **Geometric mode (ACMM)**: `compute(ref, sources, range, opts, { prior, sourceDepths })`; prior and
  source maps may have any resolution (nearest, §2 rule), a 0×0 source map = unavailable. Cost +=
  λ·Σ wᵢ·min(eᵢ, τ)/Σ wᵢ (λ 0.2, τ 3 px, forward–backward reprojection with nearest source depth).
  Runs `geometricIterations` (3) instead of `iterations`, starting from 1 % perturbation. Passing
  `{geometric:false}` with the data runs photometric; `{geometric:true}` without data throws.
  **Multi-scale ACMM** is possible with this API: photometric at ½ resolution for all views, then
  geometric at full resolution with the ½-res maps as prior + sources.
* **Output**: `cost` is always the photometric cost; depth = 0 when cost > `maxCost` (0.5) or z ∉
  [min, max]; normals unit, camera frame, facing the camera, (0,0,0) when invalid.
* **CPU ≡ GPU**: same PCG random streams (per pass and pixel), same sampling order; results do not
  depend on the band tiling (bit-identical on GPU). GPU and CPU agree statistically (≈ 70 % of pixels
  within 1e-4, all metrics within 0.2 %). The CPU reference also has an in-bounds fast path and reuses
  costs of duplicate candidates.
* **GPU scheduling (iOS)**: one command buffer in flight (submit → `onSubmittedWorkDone` → next), row
  bands adapted on line to `targetSubmitMs` (20 ms, capped by `maxPixelViewsPerSubmit` = 1.5 M pixel·views;
  disabled on software adapters); every dispatch increments a sentinel counter, and a mismatch with
  the number of submissions (work dropped while hidden) triggers up to 2 full retries, then an Error.
  Patch window + checkerboard loops are unrolled in the generated WGSL (Safari loop counters / bounds
  checks); no structs by value, no vec3 in buffers, 5 storage buffers, 8×8 workgroups, no workgroup
  memory. **All 8 pipelines (source buckets 4/8/12/16 × init/propagate) are compiled in `create()`**
  (raced against device loss + 30 s timeout → sync fallback) — create the matcher after capture, not
  while the camera preview runs. `selfTest()` compares GPU vs CPU on an 80×60 synthetic scene (~0.5 s)
  — call it once and fall back to `patchMatchCPU` if `ok` is false. Readback is chunked through a
  ≤ 4 MB staging buffer (one `getMappedRange` per map). Resources are reused (per-size buffers,
  grow-only texture arrays). Device loss ⇒ `Error('PatchMatchGPU: GPU device lost …')`.
* **Memory** (GPU): 32 B per reference pixel (+ ≤ 4 MB staging), 1 B per source pixel, + 4 B per source
  pixel in geometric mode. 1600×1200 with 8 sources: ≈ 61 + 15 (+ 61 geometric) MB; JS output
  20 B/px (+ 20 B/px transient during readback).
* **Measured** (synthetic, 1 ref + 6 sources; "surface" = GT pixels > 5 px from depth edges, viewing
  angle < 75°): CPU 160×120 near: surface 99.8 % valid, 89 % within 1 %, median normal 5°; all pixels
  71 % within 1 % (window-radius fattening/rim effects dominate at this resolution); 5.6 s (Node).
  GPU 256×192 near: surface 94 % within 1 %, all pixels 85 %, normals 4.3°; far (6.5–56 m GT, search 4–76 m): surface
  90 % within 1 %, 97 % within 2 %. Geometric pass (192×144): all-pixel accuracy 78 → 82 %, surface
  92 → 95.5 %. 640×480 / 8 sources: 97 % within 1 % after 2 iterations. SwiftShader runs ≈ 1 s per
  iteration at 256×192 — a CPU emulation, not representative of Apple GPUs.
* **Limitations / TODO**: window-based fattening at occlusion boundaries (left to fusion consistency
  checks); no multi-scale driver inside the module; hardware bilinear weights have ~8-bit sub-texel
  precision on some GPUs (≈ 1/256 px); r32float source depth arrays could be r16float to halve
  geometric-mode memory; on-device (A-series) profiling of the unrolled kernel still to be done.

### `src/features` — CPU part: review notes (adversarial review)

* **`H` guide is sign-invariant**: a homography is only defined up to a (possibly
  negative) scale (DLT/SVD estimates have an arbitrary sign). `matchGuidedCPU` orients H
  by majority vote so that most keypoints of a map with w > 0; only points mapping to /
  beyond infinity under that orientation are excluded. (Before: −H matched nothing.)
  `match-gpu` must do the same.
* **Matchers recompute descriptor norms per call** (O(128 n), < 1 ms for 4k features), so
  a descriptor buffer refilled in place can never produce stale similarities. The
  `descriptorInvNorms` / `unitDescriptors` helpers stay cached per array (treat
  descriptors as immutable once passed to them).
* **Uint8 input is resampled directly** (`resizeFloat32` accepts Uint8/Uint8Clamped and
  scales by 1/255 on the fly, streaming one row of scratch): no full-size Float32 copy
  (48 MB for a 12 MP frame) and no full-height temporary in `resizeGray`, `sharpness` or
  `extractSiftCPU`.
* `extractSiftCPU(img, opts, fullWidth)` with only one full-resolution dimension derives
  the other from the input aspect (before: the other defaulted to the input size, i.e. a
  wrong anisotropic mapping); invalid sizes throw. `rotateRGBA90` throws on non-finite
  turns. `sortByScoreDesc` ranks NaN scores as −∞ (consistent comparator).
* Conventions now pinned by tests (would not be caught by matching precision): orientation
  sign (warp by +α clockwise as displayed ⇒ θ increases by α), exact-90° equivariance
  (θ + π/2, identical descriptors), absolute descriptor layout incl. y′ sign and
  orientation-bin direction (mutation-tested: transposed bins, flipped x′/y′ and flipped
  (θ − φ) all fail), S = 2/4 localisation + DoG scale 2^(−1/(2S)), F-guided band walk
  with a finite epipole (steep + flat lines) against a brute-force reference.

### `src/math` — review pass (supersedes the conflicting points of the `src/math` notes above)

* **poly**: `polyRealRoots` no longer makes the polynomial monic (a tiny leading
  coefficient overflowed it); coefficients are normalised by exact powers of two, root
  bounds are computed from exponents, and when deg·log2(max/min root bound) > 800 bits
  the modulus range is split into power-of-two bands (each searched on its own scaling),
  so small roots survive next to huge ones (verified: cubic with lead 1e-300, degree 10
  with 1e-100, degree 20 with 1e-40). rtsafe cap 300 → 2500 (roots ~1e-90 of the bracket
  were returned unconverged). Cluster-merge noise factor 8× (64× in the final merge) →
  1.5× the rigorous Horner bound: (x−1)(x−1−5e-7)(x−5) now yields 3 roots; no 2–4-fold
  root was lost or split in 14k rounded-coefficient trials. `solveQuadratic` scales by a
  power of two (b²−4ac overflowed/underflowed), returns [] for non-finite coefficients
  and drops overflowing roots. `polyRealRootsSturm` keeps a single scaling (cross-check only).
* **NaN policy**: non-finite input now yields NaN instead of a plausible default in
  `symEig`, `mat3ToRotvec`, `mat3ToQuat`, `quatNormalize`, `axisAngleToMat3`,
  `rotationBetweenVectors`, `vec3Normalize` (zero input keeps the documented
  identity/zero), and `median`/`percentile`/`mad` (NaN data or NaN p → NaN).
  `quickselect` returns NaN for k ∉ [0, n). Normalisers are over/underflow safe (hypot).
* **Subnormal scale**: SVD/svd3/nullSpace/nullVector/qr/solveLeastSquares scaled by
  1/max|a|, which overflowed for max|a| < 2^-1024 (NaN / zero outputs); fixed.
* **Rng**: `sample` uses Floyd's permutation variant — output ORDER is now uniform too.
  Negative seeds no longer collide with small positive ones (−1−k ≡ k); streams for
  seeds in [0, 2³²) are unchanged.
* Regression tests: `src/math/adversarial.test.ts` (each test names the defect; 16 of
  its 30 tests fail on the pre-review code).

### `src/harness/synth` + `e2e/` — synthetic datasets & evaluation

* **Purpose**: rendered sequences with exact GT for headless end-to-end tests
  (`node e2e/generate-synth.mjs [object|yard|all] [--frames N] …`) and a scorer
  (`node e2e/metrics.mjs <dataset> <result.json>`). Full format, CLI and metric
  definitions: `e2e/README.md`. Datasets go to `e2e/data/<name>/` (gitignored, served by
  Vite at `/e2e/data/<name>/`); `src/harness/synth/dataset.ts` → `loadSynthDataset(url)`.
* **Ground truth is self-contained**: the generator uses its own tiny PRNG (`rng.ts`,
  sfc32) and 3×3 helpers (`mat.ts`) instead of `src/math`, so a bug in the code under test
  cannot leak into the GT. (Deliberate exception to the "always use `Rng`" rule.)
* **Trajectories** (`trajectory.ts`) are written with second-order forward-mode jets
  (`jet.ts`): positions, R_wd, dR_wd, d²R_wd, ω_dev, ω̇_dev are exact (tested against
  finite differences). The IMU (`imu.ts`) is sampled from them per §2:
  gyro = ω_dev, accel = R_wdᵀa, accelG = R_wdᵀ(a − g); camera and IMU co-located.
* **Rendering** (`render.ts`, WebGL2 via three): unlit MeshBasicMaterial scenes (Lambert
  baked into vertex colours), rendered undistorted to a half-float target at 2× over a
  window covering the undistorted sensor, then resampled per output pixel (2×2
  sub-samples) through the inverse RADIAL distortion (Newton). The OpenCV pose and
  intrinsics map to a three camera with an off-axis frustum (`three-camera.ts`), so the
  principal point is exact. A GPU test checks that marker centroids land within 0.05 px
  of their RADIAL projections. Noise, vignetting and exposure are applied in linear light
  before sRGB encoding.
* **Time base**: frames report `t = tTrue + timeOffsetMs` (default +25 ms); IMU, GPS and
  compass are on the true clock. Frames, depth maps and keypoints are in the upright UI
  orientation (θ = 0 portrait for object, θ = 90 landscape for yard).
* **GT surface**: area-uniform samples with int8 normals and per-sample visibility
  counts (depth-tested against every frame). The object scene has an extra dense
  "focus" set on the object of interest, so mm-level accuracy can be measured.
* **Known limits**: global shutter (no rolling shutter); no lighting changes beyond
  ±5 % exposure; no specularity; the camera and IMU have no lever arm; textures repeat
  only where the material is periodic (bricks, shingles, grass/soil detail tiles of
  2.3 / 3.7 m). SwiftShader renders about 0.75–0.9 s per frame, so a full 240-frame
  yard dataset takes about 4 min.

### `src/dense` — PatchMatch review notes (corrections to the PatchMatch subsection above)

* **Multi-scale recipe (correction)**: do NOT pass ½-resolution maps as `sourceDepths` of a
  full-resolution geometric pass. Their reprojection error (λ·min(e, τ) in full-resolution pixels)
  dominates the photometric term. On the near scene this gives 62 % within 1 %, against 88 % for
  plain full-resolution photometric. Use instead:
  1. photometric at ½ resolution (all views);
  2. `compute(ref, srcs, range, { geometric: false, iterations: 2 }, { prior: halfMap, sourceDepths: [] })`.
     This is a **prior-initialised photometric pass**, new semantics for `{geometric:false}` + data,
     which used to ignore the data. It matches 6 random-init iterations (88.2 % vs 87.5 %) at about
     40 % of their cost;
  3. geometric pass with the **full-resolution** maps as prior and sources.

  Any pass initialised from a prior starts at `geometricPerturbation`. `priorToPlanes`
  re-intersects resampled planes with the target pixel ray, which is exact for planes; before, the
  depth was copied, with errors up to 2 % on slanted planes.
* **`selfTest()`** now runs the CPU reference with the instance's `create()` options. Before, a GPU
  instance created with non-default options was compared against a different algorithm and
  reported `ok: false` (agreement 0.59), forcing the CPU fallback.
* **Error scopes**: they are opened and closed synchronously around each encode + submit, and never
  held across `await`. The scope stack is shared by every user of the device in the realm.
  `onSubmittedWorkDone` is raced against `device.lost`, so a lost device cannot hang `compute()`.
  If set-up fails (e.g. out of memory), the cached resources are released. Texture arrays only
  grow, per dimension.
* **Known limits**:
  * No mip selection for sources that see the surface at a different scale. At 2× resolution,
    within 1 % drops from 81 % to 70 %; at 0.5×, to 33 %.
  * A zero-baseline source, photo-consistent at every depth, is harmless when there are at least
    2 informative sources, but it lowers the reported cost. `selectSourceViews` must enforce a
    minimum triangulation angle.

### `src/sensors` (camera, IMU, compass/attitude, GPS, orientation, permissions, FramePump)

* **API superset of §4.7** (documented signatures kept). Files: `frame-orientation.ts`,
  `normalize.ts` (pure, DOM-free), `motion.ts`, `orientation.ts`, `geolocation.ts`, `camera.ts`,
  `wakelock.ts`, `permissions.ts`, `diagnostics.ts`, `frame-pump.ts`, `index.ts`.
* **θ / R_cam_device**: `screenAngle()` prefers `window.orientation` ((wo+360)%360; 90 = device
  turned counter-clockwise, top edge LEFT) and falls back to `screen.orientation.angle` — on iPad
  WebKit reports `screen.orientation.angle = (90 − θ) mod 360` (natural = landscape), so the
  angle API alone is wrong there. `cameraFromDevice(θ)` = diag(1,−1,−1)·Rz(θ), exact integers.
  `canonicalRotation(θ_s, θ_f)` = ((θ_f − θ_s)/90) mod 4 CLOCKWISE quarter turns (same
  convention as `rotateRGBA90`); afterwards `R_cam_device = cameraFromDevice(θ_s) =
  quarterTurnMat3(k)·cameraFromDevice(θ_f)`, pixels map with `rotateImagePoint` (= rotateRGBA90),
  a world→camera pose R becomes `quarterTurnMat3(k)·R`. `ScreenAngleTracker` records θ changes;
  frames < 500 ms after a change are dropped (Safari 26 transient mis-rotation).
* **IMU normalisation** (normalize.ts): rotationRate alpha/beta/gamma = ω about device x/y/z
  (deg/s → rad/s; NOT alpha = z), `interval` s→ms, spec-sign accelerations. The iOS sign
  inversion is detected at run time (prior −1 only when `DeviceMotionEvent.requestPermission`
  exists): check B compares the raw specific force with the DeviceOrientation up-vector
  (β, γ) at near-stillness (decides after 30 consistent votes); fallback B′ uses the UI-up axis
  (≥ 2 s of votes); B″ measures the sign of `acceleration` relative to the gravity-including field;
  D estimates the gyro signed permutation + units from orientation-derived body rates. The ring
  buffer stores RAW values, so a calibration change re-signs everything still buffered
  (`calibration.version` increments; drained samples are not revisited). First (replayed) event
  after attaching and bit-identical duplicates (only for unquantised data) are dropped.
* **Timing**: `ImuSample.t` = `event.timeStamp` except for bursts after a main-thread stall
  (arrival gaps < 0.35 × nominal): the run is re-spaced at the nominal interval ending at the
  newest arrival (never moved later; drained samples frozen). The ~10–20 ms platform latency is
  NOT removed (estimate a camera–IMU offset). Camera `t` = rVFC `captureTime` (post-ISP arrival,
  ~10–40 ms after exposure) else `presentationTime` else callback time (`meta.timeSource`).
* **integrateGyro** returns R_{d(t1)←d(t0)} = R_wd(t1)ᵀR_wd(t0): cubic Hermite ω(t) (3-point
  parabola tangents; linear at data ends / gaps > 100 ms; held outside the data) + 4th-order
  Gauss–Magnus steps split at sample times. Error on smooth hand-like motion at 60 Hz with ±2 ms
  jitter < 3e-5 rad over 2.5 s (linear + trapezoid: ~1.4e-3). `MotionSensor.rotationBetween(t0,t1)`
  does the same on the buffer; `samplesBetween(t0, t1, { pad: true })` adds the bracketing samples.
* **MotionSensor extras**: `needsPermission()`, `lastPermissionError` ('NotAllowedError' = no
  gesture), `calibration`/`onCalibrationChange`, `gravityAt(t)` (unit DOWN, device frame: from
  DeviceOrientation, else mean specific force), `stats`, `handleMotion/handleOrientation` (replay),
  `{ target }` option (tests). `angularSpeed(t)` is NaN without a sample within 100 ms.
* **CompassSensor**: `headingAt(t)` = platform heading (iOS `webkitCompassHeading`, MAGNETIC,
  interpolated on the circle; accuracy = `webkitCompassAccuracy`; < 0 or > 25° ignored) or, with
  `deviceorientationabsolute`, the azimuth of the same reference (device top; camera axis when the
  top is > 45° from horizontal), nominal accuracy 20°. `attitudeAt(t)` = R_world_device (ENU when
  `absolute`): Chrome absolute events directly; iOS arbitrary-yaw α rotated by the estimated yaw
  offset ψ (`heading = azimuth_arb + ψ`, circular mean weighted by accuracy, τ = 30 s, motion
  > 57°/s rejected). `setDeclination(deg)` turns magnetic into true north. A "ui-top" heading model
  (iOS 26 landscape compensation, unverified for Safari) is auto-selected only when it is clearly
  more consistent across UI rotations. `cameraHeadingAt(t)` = azimuth/elevation of the optical axis.
* **CameraCapture**: label classification (English AVFoundation names + localised keywords
  + capabilities: `facingMode`, `zoom.min < 1` ⇒ virtual); `chooseCamera`: physical wide >
  unidentified rear (Android) > virtual (zoom 1, `lensMaySwitch`) > unknown > ultra-wide > tele.
  Formats: exact LANDSCAPE sizes (+ `resizeMode: 'none'` where supported ⇒ native formats only,
  no silent crop), then ideal, then device-only. 'max': 4032×3024, 3264×2448, 2592×1944,
  1920×1440, 3840×2160, 1920×1080 (filtered by `getCapabilities()` max); 'balanced': 1920×1440,
  (1600×1200), 1920×1080; 'fast': 1280×960, 1280×720 — on Apple 1920×1440 (no native 1280×960;
  smaller 4:3 presets are cropped) with `frameMaxSide` 1280 (FramePump downscales on grab).
  **`hfovPriorDeg` is the FOV across the LONG image side** (Apple main 67.3°, Pro 69.5°,
  16:9 ×1.016, cropped presets narrower, ultra-wide 104°, tele 25°, front 70°, unknown 63°);
  use `focalPriorPx = (max(W,H)/2)/tan(hfov/2)` (in the result) — valid for any UI rotation.
  Image settings (zoom 1, continuous focus/exposure/WB, torch off) are applied as separate
  advanced sets merged with the stream constraints (applyConstraints replaces everything) and
  verified via getSettings(). `lockFocus/lockExposure` work on Chrome/Android only (false on iOS);
  `lockWhiteBalance()` works on iOS — call it ~1 s after start. Interruptions: mute ⇒
  'interrupted'; unmute or re-open ⇒ `segment++` (FramePump passes it on: start a new SfM
  segment / relocalise). Hidden: optional release; on visible re-open if still muted after 1.2 s
  or ended (≤ 3 restarts / 10 s). Keep the `<video>` attached and visible (rVFC / iOS decode).
* **FramePump** delivers `PumpFrame` = ENGINE `FrameInput` + `index`, `width/height`,
  `timeSource`, `R_cam_device`, `imuCalibrationVersion`, `attitude`, `gravityDevice`, `segment`.
  Rate limit uses half a frame of tolerance (30 fps → exactly 10 Hz); never more than one frame
  in the consumer; skips while hidden, after θ changes and when the bitmap aspect contradicts θ
  (transient, re-baselined after 1.5 s). The consumer owns the bitmap (closed by the pump only if
  the consumer throws). `geo` = interpolated fix (or nearest within 30 s).
* **GeoSensor**: `watchPosition({enableHighAccuracy: true, maximumAge: 0})`; `t = timestamp −
  (Date.now() − performance.now())` at delivery; `interpolate(t)` (≤ 5 s gaps, antimeridian-safe),
  `latest`, `status`, `approximate` (accuracy > 1 km ⇒ iOS "Precise Location" off), `onFix`.
  **`GeoFix.alt` is orthometric (MSL/EGM2008) on iOS**, not WGS84-ellipsoidal.
* **requestCapturePermissions()** must be called synchronously in a click/touchend handler: it
  calls `DeviceMotionEvent.requestPermission()` before returning, then (awaited) orientation,
  camera probe (yields device labels), geolocation; never rejects; `motion: 'needs-gesture'` when
  called outside a gesture. `queryPermissionStates()` (camera/geolocation; motion is not queryable).
* **collectDiagnostics({camera, motion, compass, geo, gpu: 'device' | 'adapter' | false})**: UA,
  standalone, COI, cores, deviceMemory, screen/θ, feature flags (rVFC, MSTP, VideoFrame, WebGPU,
  wake lock, SIMD…), GPU info/limits/features (`'device'` uses `getGPU()` = creates this realm's
  device), cameras, live sensor state, storage estimate, Lockdown-mode suspicion.
* **Tests**: unit (Node) — θ/R_cam_device/canonical rotation, Euler round trips, simulated iOS and
  spec events (flat, upright portrait, landscape, tilted) incl. sign/axis/unit self-checks,
  integrateGyro vs analytic rotation, burst re-timing, compass ψ estimation, geo interpolation,
  label classification (en/de/fr/es/it/pt/ja/zh/ko/ru, Android), FramePump back-pressure;
  browser (`sensors.gpu.test.ts`, Chromium fake camera) — start/onFrame/grabFrame/controls/stop,
  'max' format choice, restart segments, start/stop race, FramePump with synthetic DeviceMotion
  events, diagnostics, wake lock.
* **Unverified on device / limits**: localised labels beyond English/German are best-effort
  keyword lists (unknown locales fall back to capabilities + rank); the compass reference when
  the phone is upright and the iOS 26 landscape heading behaviour; createImageBitmap resize
  options on Safari (falls back to native size — always use bitmap.width/height); whether
  `<video>`-derived bitmaps are rotated exactly as the UI on every iOS version (aspect check +
  hand-eye check in the engine recommended); rVFC `captureTime` latency; MSTP worker path
  (research ladder A) is not implemented here — the FramePump uses `createImageBitmap(video)`.

### `src/ba` (bundle adjustment)

* **Files**: `problem.ts` (types, defaults, `validateBAProblem`), `loss.ts`, `camera-model.ts`
  (RADIAL residual + analytic Jacobians, kept local to ba — must stay identical to §2),
  `priors.ts`, `linear.ts` (block-sparse symmetric matrix, skyline Cholesky, block-Jacobi PCG),
  `solver.ts`, `index.ts`; `synthetic.ts` = deterministic problem generator for tests /
  benchmarks (orbit = close range, strafe = 5–100 m scenes; NOT re-exported from the index).
* **API superset of §4.3** (documented signatures kept). `BAResult` extras: `successfulIterations`,
  `linearSolver` ('dense_schur' | 'pcg_schur' | 'none'), `numReducedParams`, `numFreePoints`,
  `pcgIterations`, `finalLambda`; `termination` is the `BATermination` union. Extra functions:
  `evaluateBACost(problem, opts) → { cost, rmsPx }`, `computeBAGradient(problem, opts)` (gradient
  per pose [δω, δt], point, camera [f, cx, cy, k1, k2]; 0 for fixed/pinned), plus the low-level
  pieces (`radialResidualJacobian`, `so3LeftJacobianInverse`, `SkylineCholesky`, `pcgSolve`, …).
* **Cost** = ½Σρ(‖w·r‖²) + ½Σ‖prior residual‖² (Ceres convention), r = predicted − observed px,
  w = `obsWeight` (1/σ). ρ is applied to the WEIGHTED residual, so `lossScalePx` is in pixels for
  unit weights and in σ units for 1/σ weights. Priors are not robustified. RMS values are
  unweighted, over all valid observations.
* **LM** (Ceres / Nielsen): D = clamp(diag(JᵀJ), 1e-6, 1e32) on U and V; accept if gain ratio
  > 1e-3, then λ ·= max(1/3, 1 − (2ρ−1)³), else λ ·= ν, ν ·= 2; λ ∈ [1e-16, 1e32], above ⇒
  'min_trust_region_radius' (usable). The gain ratio uses the exact UNDAMPED quadratic model
  (valid for inexact PCG steps and frozen points). Robust losses by IRLS (√ρ' scaling — what Ceres
  does for ρ'' ≤ 0): linear convergence (≈ 5–15 iterations), trivial loss converges quadratically.
  Defaults: `functionTolerance` 1e-6 (relative cost decrease), `gradientTolerance` 1e-10 (max |g|),
  `parameterTolerance` 1e-8 · (‖x‖ + tol) with ‖x‖ over ALL free parameters incl. points (pass 0
  to disable on large scenes). `success` is false only for 'invalid_initial_cost' and
  'linear_solver_failure'; the problem always ends at the last accepted state (also on cancel,
  time limit, or an exception from `onIteration`).
* **Gauge / degeneracies**: `translationFixedMask` components are zero Jacobian columns with unit
  diagonal in S ("pinned"), dropped from the direct solve. Fixed points only feed U. A free point
  whose damped 3x3 V fails a relative-pivot test (1e-12) is frozen for that step (δX = 0, no Schur
  term); points without valid observations are untouched. Observations with a non-finite residual
  at the input (or NaN / negative / infinite weight) are ignored for the whole run; a trial step
  that makes a residual non-finite is rejected. |Z| is clamped to ≥ 1e-12 (points behind a
  camera give finite mirrored residuals that robust losses down-weight).
* **Priors**: centre prior Jacobian w.r.t. δω is −Rᵀ[t]×/σ, w.r.t. δt −Rᵀ/σ. Relative-rotation
  prior uses the EXACT Jacobian (inverse left Jacobian J_l⁻¹(φ)), not the first-order one.
* **Linear solver**: reduced system over [free poses (6 each, pinned components kept as decoupled
  unit rows) + free intrinsics per camera]; block order = free poses in INPUT order, then cameras
  (so intrinsics rows, which couple to all their poses, do not widen the envelope). S is assembled
  explicitly block-sparse (pattern fixed per call; per-point pair offsets precomputed, no
  allocation inside iterations). **'dense_schur' = Cholesky in packed skyline (envelope) storage**
  over the fixed row profile of S: same result as a dense Cholesky, memory/work only inside the
  profile. **Deviation (extension) of 'auto'**: direct when n ≤ 2400 (as documented) OR when the
  envelope is small (≤ 4M entries and ≤ 3e8 multiply-adds — e.g. sequential captures), else PCG
  (block-Jacobi on S, x₀ = 0, ‖r‖ ≤ `pcgTolerance`·‖b‖). Callers (sfm) should pass poses roughly
  in capture order — the envelope (and thus the direct-solver cost) depends on it; a random order
  just falls back to PCG. A forced 'dense_schur' falls back to PCG above 18M envelope entries.
* **Memory**: no per-observation Jacobians are stored. Per call: W = 18 doubles per (point,
  distinct camera-side block) ≈ 144 B × (#obs + #points), Schur pair offsets 4 B × Σ m(m+1)/2
  (m = blocks per point), U and S (block-sparse), V/V⁻¹ 12 doubles per point, 2 parameter copies.
  ≈ 85 MB for 400k observations / 60k points; ≈ 20 MB for 100k / 20k.
* **Performance** (Node 22, x86, on a loaded 4-core machine; `solver.perf.test.ts` logs them):
  100 poses / 20k pts / 98k obs, direct (n = 596): ≈ 140–210 ms per LM iteration (first call incl.
  setup + JIT ≈ 600 ms); 500 / 60k / 406k: PCG ≈ 850 ms/iteration (≈ 440 PCG iterations/solve —
  block-Jacobi is weak on long chains), 'auto' (skyline) ≈ 570 ms/iteration with exact steps;
  local BA (12 free + 18 fixed poses, 49k obs) ≈ 50 ms/iteration after warm-up. Hot loops: Schur
  6x6 pose blocks unrolled with W in registers, linearisation unrolled for the pose block.
* **Accuracy** (tests): self-calibration from f +10 %, k1 +0.05, perturbed poses/points → f error
  1.3e-5, RMS = LS noise floor (0.658 vs 0.653 px expected, σ = 0.5 px); 10 % gross outliers with
  Cauchy → f error 3e-5, inlier RMS 0.647 px (trivial loss: f error 12.8 %); depth range 2 m–2 km
  in one scene converges; points at 1–1000 km stay finite. Dense and PCG solutions agree to 1e-7.
* **Limitations / TODO**: IRLS only (no Triggs correction, like Ceres for concave ρ); points far
  beyond the baseline converge slowly along the ray (no inverse-depth parameterisation; harmless
  for reprojection); PCG uses block-Jacobi only (no visibility-based preconditioner); synchronous
  and single-threaded (run it in the engine worker; use `timeLimitMs` / `cancel` for live BA).

### `src/io` — review notes (adversarial review; supersedes conflicting points of the `src/io` notes above)

* **SessionStore reconnects** (research §2.5 rule): a transaction that cannot start
  (InvalidStateError), or that aborts with InvalidStateError / UnknownError while the
  connection can no longer start transactions (probe), is retried ONCE on a freshly opened
  connection. This covers WebKit "Connection to Indexed Database server lost" and our own
  close on `versionchange` (another tab upgrading or deleting the DB; a deleted DB is
  recreated empty). An aborted transaction committed nothing, so the retry is safe. `close()`
  is final. `store.db` is now a getter (current connection).
* **Blob → ArrayBuffer fallback** becomes permanent only if the buffered retry SUCCEEDS.
  Before, a DataCloneError from a non-cloneable frame `info` switched the store to buffer
  mode for good.
* **saveFile**: in an iOS Home Screen app (`navigator.standalone === true`), a failed share
  (typically NotAllowedError after an expired activation) is **rethrown** instead of falling
  back to `<a download>`, which is broken there (WebKit 275288 / 290847). The UI should ask
  for another tap. `sanitizeFilename` keeps the extension when shortening to 200 units and
  never splits surrogate pairs.
* **COLMAP export validates**: ids are non-negative integers, width/height positive
  integers, every image references an exported camera, and every track element references an
  exported image and an existing POINTS2D index (RangeError otherwise). POINTS2D entries
  referencing a point id that is not exported (e.g. filtered points) are written as −1, and NaN
  colours as 0. COLMAP crashes or mis-parses on all of these.
* **ZIP**: `readZip` accepts plain archives with exactly 65 535 entries (ZIP64 only when the
  locator is present) and is not fooled by an EOCD signature inside the comment. Writers refuse
  such comments. `'.'` path segments are dropped, so duplicates are detected. Non-Uint8 views
  are stored as raw bytes.
* **PLY**: element counts are checked against the remaining body before any allocation (a
  hostile header otherwise triggers multi-GB allocations, i.e. jetsam). A trailing empty polygon
  in the binary face fast path no longer throws a DataView RangeError.
* **GLB**: an empty scene omits `scene.nodes` (schema: minItems 1).
  **serialize**: `float32ToFloat16Bits` is correctly rounded for any double (round-to-odd into
  float32), mesh indices use u16 only when every index ≤ 65535, and JSON-safe encoding keeps −0
  and throws on Set / Date / ArrayBuffer / DataView / Blob instead of emitting `{}`.
* Tests: `src/io/adversarial.test.ts` (Node) and `src/io/adversarial.gpu.test.ts` (Chromium).
  `zip.test.ts` / `serialize.test.ts` no longer need `@types/node`; the main tsconfig is
  browser-typed and `src/io` typechecks clean.

### `src/viewer` — review notes (adversarial review; amends the `src/viewer` notes above)

* **Fixed**: CPU picker now honours the confidence filter (`PickSplatModel.confidence/minConfidence`,
  same `conf < min` test as the shader) and intersects meshes exactly (`raycastTriangles`,
  Möller–Trumbore, ≈ 20 ms / 1 M triangles) instead of treating vertices as 1 px points (taps inside
  large triangles missed; points behind a mesh were picked). `setPointCloud` with the same
  positions/colours but NEW `radii`/`confidence` arrays uses them (same byte size → re-upload into the
  existing buffer, else the layer is rebuilt; in-place edits of arrays are still not detected — pass
  new arrays). `fitToContent()` while the container is 0×0 is deferred to the first real resize (the
  aspect was unknown → wrong framing after the app shows the viewer). Context restore re-schedules a
  frame (animations/damping no longer stall). WebGL creation failure removes the canvas and restores
  the container before rethrowing; `dispose()` restores the container's inline `position`.
  `screenshot()` rejects while the context is lost (was a blank PNG). Degenerate content (all points
  coincide, no grid/cameras) no longer gets a zero-thickness near/far range. A pinch between two taps
  no longer counts as a double tap. GPU-pick sample selection is O(m) (was O(want·m)).
* **iOS**: canvas also sets `user-select: none`, `-webkit-touch-callout: none`,
  `-webkit-tap-highlight-color: transparent` and cancels Safari `gesturestart/gesturechange`.
* **Verified by tests** (`viewer.review.gpu.test.ts`, `review.test.ts`): (x,y,z)→(x,z,−y) mapping of
  targets/content/projection; splat diameter = 2·r·f_px/z (± 1.5 px) and fixed CSS sizes at DPR 1/2;
  overlays behind EDL-composited points are depth-occluded; frustum orientation (rows of R, C = −Rᵀt,
  up marker → world +Z); picks/measures return WORLD coords under `setWorldUp('auto')`, at DPR 1.5/2,
  at canvas edges, at 100 m scale with km outliers (< 0.5 % distance error), during LOD; mouse drag +
  damping terminates; no growth of GL buffers/textures/FBOs/VAOs or scene children over repeated
  content updates; transparent background alpha.

### `src/features` — GPU part (SiftGPU, MatcherGPU, createFeaturePipeline)

* **Files**: `sift-gpu.ts` (+ `shaders/sift-wgsl.ts`), `sift-gpu-host.ts` (WebGPU-free host logic, Node-tested:
  resize tables, candidate decode / order / de-dup, describe order, assembly, self-test image, `featureParity`),
  `match-gpu.ts` (+ `shaders/match-wgsl.ts`), `match-gpu-host.ts` (norms + content hash, guide records, Morton
  order, tile boxes, exact top-2 reduction, `emitMatches`), `gpu-common.ts` (`GPUFeatureError`,
  `GPUWorkDroppedError`, `ParamBlock`, `Staging`, `BudgetEncoder`, `WorkStamp`), `extractor.ts`.
* **API superset of §4.4**: `SiftGPU.create(ctx, opts?, cfg?: SiftGPUConfig)`, `extract`, **`extractFromBitmap(bitmap |
  ImageData | canvas, fullW?, fullH?)`** (copyExternalImageToTexture → Rec.709 luma of the sRGB values (= rgbaToGray)
  + area resampling in one kernel, no CPU pixel work), `selfTest()`, `memoryBytes()`, `releaseCaches()`,
  `lastTimings`, `lastProfile`. `MatcherGPU.create(ctx, cfg?)`, `match`, `matchGuided`, `selfTest()`, `clearCache()`,
  `cacheMemoryBytes()`. `FeaturePipeline` adds `fallbackReason` and `extractFromBitmap`; `createFeaturePipeline`
  options add `context` (own GPUContext, `null` = no GPU), `siftGPU`, `matcherGPU`, `onFallback`, `selfTest` (default
  true). Create the pipeline once at start-up: 19 pipelines are compiled up front (blocks the queue on Safari).
* **CPU equivalence (SIFT)**: same plan / scale space / kernels (`gaussianHalfKernel` weights uploaded) / DoG /
  26-neighbour strict extrema / Taylor refinement with moves / contrast + edge tests / orientation (36 bins, 6
  smoothings, peaks ≥ 0.8) / VLFeat descriptor / RootSIFT quantisation, all in f32 on the GPU. GPU candidates are put
  back into the CPU raster order (octave, start level, row, column) before the `seen` de-duplication, and
  preselection, describe order (`order` comparator), `octaveToFull` and the final `selectFeatureIndices` run on the
  CPU with the CPU code ⇒ identical FeatureSet layout and ordering semantics. Measured on 640×480 textures: identical
  feature counts, 100 % of CPU keypoints reproduced (max position difference 4e-3 px), median descriptor cosine
  1.00000, ~70 % of descriptors byte-identical, the rest differ by ±1 in a few bytes (f32 vs f64), warped-pair
  matching precision identical to the CPU pipeline.
* **SIFT GPU design**: per octave one buffer with all S+3 Gaussian levels (DoG formed on the fly); blurs are
  separable — shared-memory row/column tiles on hardware adapters, direct loads on software adapters (`cfg.blur`;
  SwiftShader's barrier emulation makes tiles ~10× slower; both bit-identical); detection appends 8-word records
  through an atomic counter with capacity guard (grow + re-run; counter + first `1.25 × last count` records read in
  one round trip); orientation = one 64-thread workgroup per keypoint with per-thread partial histograms in
  workgroup memory (no float atomics); descriptors = one 64-thread workgroup per (keypoint, orientation) slot,
  trilinear accumulation with **fixed-point u32 workgroup atomics** (scale `descriptorFixedPoint(σ, S)` = 2^22 for
  defaults, overflow-safe for 0..1 images), RootSIFT + quantisation + byte packing on the GPU. Work is split into
  command buffers by a cost budget (`submitBudget`, ≈ pixel-taps). Round trips per extraction: 2 (3 if the candidate
  guess is too small).
* **Matcher design**: GEMM-style kernel — 16×16 threads own 64 rows × walk 64-column tiles; 16-dim chunks of both
  tiles are unpacked (manual shifts) into workgroup memory as f32; each thread accumulates a 4×4 block of exact
  integer dots (< 2^24 ⇒ exact in f32), keeps a per-row top-2 (f32 score dot·invNorm, ties → smaller index), merged
  over the 16 column groups in workgroup memory; output per (row, column chunk): (index, dot) of the best two. The host
  recomputes these similarities **exactly in f64** as match.ts and applies ratio / maxDistance / mutual with the same
  logic ⇒ results identical to matchCPU / matchGuidedCPU in all tests (100.00 % overlap, incl. real 4096×3366 SIFT
  sets). Both directions run on the GPU for the mutual check (reverse skipped when `mutual: false`). Guided: `H` →
  predictions exactly as matchGuidedCPU (sign vote), `positions` (NaN = unconstrained, ±Inf / > r + 1 px outside b =
  never), `F` → unit epipolar lines in b-centred coordinates; the predicate is evaluated per pair and whole column
  tiles are culled (rows and columns Morton-sorted, conservative tile boxes +1 px) — 2–4× fewer pairs on warped pairs.
  Descriptor buffers (packed u32) + f32 inverse norms are cached per FeatureSet (WeakMap + LRU byte budget, default
  64 MiB, + FinalizationRegistry); a content hash is checked on every call (in-place refills are re-uploaded).
  Known (negligible) difference: exact f64 ties in guided mode resolve to the smaller index (CPU: grid-walk order).
* **Robustness**: input errors are plain `Error`s; GPU failures are `GPUFeatureError` (device loss — raced against
  every wait —, validation / OOM via error scopes, readback failures, unsupported sizes/radii). Every command buffer
  starts with a **work stamp** (attempt id + executed command buffers, `WorkStamp`), checked in every readback: work
  dropped by iOS while hidden ⇒ automatic retry (`maxAttempts` 3), then `GPUWorkDroppedError` (transient). The pipeline
  falls back to the CPU permanently on `GPUFeatureError` / device loss (warn once, `onFallback` once, the failed call
  is redone on the CPU), and per call only on `GPUWorkDroppedError`. `selfTest()` (synthetic 128×96 image / random
  descriptor sets vs the CPU) runs at pipeline creation; a mismatch selects the CPU backend.
* **Memory (GPU)**: SiftGPU per cached processed size (`maxCachedSizes` 2): (S+3) planes × Σ octave pixels × 4 B +
  one temp plane — 1600×1200: ≈ 69 MB; 640×480 (upsampled 1280×960): ≈ 44 MB; plus the input (u8: 1 B/px, f32: 4 B/px,
  bitmap: RGBA8 texture 4 B/px; f32 input at the processed size is written straight into the pyramid) and ≈ 3 MB fixed
  (candidates 65 536 × 32 B, keypoints / orientations / descriptors for maxFeatures × maxOrientations). Octave buffers
  above the device binding limit throw (lower maxImageSize). MatcherGPU: descriptor cache budget + 16 B × (na + nb) ×
  chunks per call.
* **Measured (SwiftShader, informational only — CPU emulation)**: 640×480 (octave −1) ≈ 2.4–4.4 s, 320×240 ≈ 0.6 s;
  matching 2000×2000 ≈ 2.2 s, 4096×3366 ≈ 3.4 s. `cfg.profile` records per-kernel times for on-device profiling.
* **TODO / limits**: no on-device (A-series) profiling yet — tile sizes, submit budget and the tiled-vs-direct blur
  choice should be tuned on an iPhone; descriptors for slots dropped by the final selection are still read back
  (≤ 128 B × maxFeatures × maxOrientations); the matcher keeps the top-2 per column chunk (a third candidate tying
  within f32 rounding could in principle swap — never observed).

### `src/ba` — review notes (adversarial review; supersedes conflicting points of the `src/ba` notes above)

* **Verified independently** (`src/ba/review.test.ts`): the LM step equals a brute-force solve of the full damped
  normal equations (no Schur) to < 1e-7 relative, for every loss, dense and PCG, λ up to 5, with weights, masks,
  fixed poses/points, 1–3 cameras with mixed refine flags, centre + relative-rotation priors (incl. swapped block order
  and fixed partners), plus 12 randomised configurations (80 fuzzed once). The λ update matches the exact undamped
  model decrease (tested at ρ ≈ 0.5). Frozen (rank-deficient) point blocks give exactly the "point held fixed" step.
* **Fixes**: soft_l1 ρ used `2b(√(1+s/b) − 1)`, which cancels to 0 for s/b ≲ 1e-16 (wrong costs / gain ratios with
  large loss scales) — now `2s/(√(1+s/b)+1)`; Cauchy / soft_l1 stay finite when s/b overflows. The loss scale b = a² is
  clamped to [1e-150, 1e150] (a tiny or huge `lossScalePx` produced NaN costs ⇒ 'invalid_initial_cost').
  Observations whose WEIGHTED squared residual overflows at the input are ignored like non-finite ones (one garbage
  keypoint no longer aborts the run). A step below `parameterTolerance` is now still taken when it lowers the cost
  before stopping (Ceres discards it). An already-set `cancel` token returns before the structure build. The PCG
  preconditioner stores only the diagonal blocks (it allocated a full copy of S).
* **Caveats for callers**: priors are not robustified — gate GPS centre priors (multipath outliers) in `src/sfm`, or
  use a large σ. `parameterTolerance` / ‖x‖ depends on the world origin and on the number of points; for the final
  global BA prefer `parameterTolerance: 0` with `maxIterations` as the bound (COLMAP-style). Explicit Schur cost and
  pair memory grow with Σ m² (m = views per point): tracks of ~30 views at 500k observations ≈ 1.3 s / iteration on
  desktop Node with ~85 MB extra.

### `src/geometry` (camera model, poses, LO-MSAC, minimal solvers, two-view, PnP, triangulation, Sim3)

* **API superset of §4.2** (all documented signatures kept). Extras: `normalizedToPixel`, `undistortPixel`,
  `intrinsicsMatrix[Inverse]` (camera.ts); `poseFromMat34` (pose.ts); `ransacIterations`, `maskToIndices`,
  `RansacOptions<M>` (generic default param) with `minInlierRatio` (iteration cap, COLMAP semantics),
  `initialModels` (seed hypotheses), `loIterations`, `loThresholdMultiplier`; estimator extensions `refine`
  (LO from the current model, e.g. nonlinear), `minRefitSize`, `residualsSqRange` (ransac.ts);
  `essentialLinearIdx`, `projectToEssential`, `essentialFromPose`, `cheiralityCount`, `CHEIRALITY_MAX_DEPTH`,
  `sampsonDistancesSq[Range]`, `refineRelativePose` (LM on Huber/Cauchy Sampson error, R left-multiplicative,
  t on S²), `tangentBasis` (essential.ts); `fundamental8ptIdx`, `enforceRank2`, `fundamentalFromEssential`;
  `homographyDLTIdx`, `homographyTransferErrorsSq[Range]`, `homographySampleDegenerate`, `decomposeHomography`
  (Ma–Soatto–Košecká–Sastry, 4 solutions, sign fixed by correspondences); `RefinePoseOptions`,
  `PnPRansacOptions`; `TwoViewOptions`, `TwoViewGeometryResult` (adds `numInliersE/F/H`,
  `planarAlternative`), `Essential/Fundamental/HomographyEstimator`, `rotationFromCorrespondences` (Kabsch on
  bearings); `AlignSim3Options` (`confidence`, `maxIterations`, `fixedScale`); `eigenvaluesNonsymmetric`,
  `eigenvectorInverseIteration` (eigen.ts). `internal.ts` / `testutil.ts` are not exported.
* **LO-MSAC**: MSAC cost Σ min(r², T²); every new best model is locally optimised (4 refits on inliers with
  the threshold shrinking 2T → T, accepted only if the cost drops; Lebeda et al. 2012), plus a final pass
  at T. Adaptive iterations log(1−p)/log(1−wˢ) from the best inlier count. With `residualsSqRange`, scoring
  is blocked (128) with an **exact** bail-out once the partial cost exceeds the best (results unchanged,
  ~2× fewer residual evaluations). Estimators must return fresh model objects from `fit`.
* **5-point**: Stewénius et al. 2006 action matrix (GRevLex template, 10×20 Gauss–Jordan, action matrix of
  ℓ = 0.3x + y + 0.7z, balanced Hessenberg/Francis-QR eigenvalues, inverse-iteration eigenvectors) + Gauss–
  Newton polish on the 10 cubic constraints; near-real complex eigenvalue pairs (|Im| ≤ 0.25(1+|Re|)) are
  polished and kept only if they converge to a real root. Nistér's degree-10 polynomial was implemented
  first and rejected: its det(B(z)) coefficients cancel catastrophically for small baselines (true E lost in
  26 % of noise-free samples at baseline/depth 1e-3…1e-2). Measured (noise-free): 100 % for baseline/depth
  ≥ 0.01, 98–99 % at 1e-3…1e-2, ~75 % at 1e-4…1e-3 (near-pure rotation; such samples carry no translation
  information anyway); ~45–60 µs/solve warmed (Node x86), ~4.9 solutions on average.
* **Two-view** (`estimateTwoViewGeometry`, COLMAP model selection, details in the JSDoc): after E-RANSAC the
  pose is refined twice — Huber(T) LM on the Sampson error, then the cheirality-consistent pose is refined
  with a Cauchy kernel at σ̂ = 1.4826·median|r| on the inliers *in front of both cameras*. Reason: random
  outliers that happen to lie within T of their epipolar line have huge parallax and therefore large
  leverage on t at small baselines (scene 50 m / baseline 1 m: t error med 1.9° → 0.4°, max 5.5° → 1.0°);
  about half of them triangulate behind a camera. F is seeded with the F implied by E; H budgets only
  need to reach the ratio that can change the decision. Defaults: maxErrorPx 4, confidence 0.999,
  minNumInliers 15, minInlierRatio 0.25, E/F ≥ 0.95, H/(E|F) > 0.8, panoramic below 0.5° median
  triangulation angle. 'planar' R, t come from the H decomposition (2-D constraint per point, more
  accurate than E) on the branch selected by E; the other branch of the planar two-fold ambiguity is
  `planarAlternative` (it cannot be resolved from two views — the mapper should try both with a third
  view). 'panoramic': R = Kabsch rotation fit refined on its own inliers, t = (0,0,0), inliers = H inliers.
  `rotationPrior` (gyro) is used only to seed H with K₂RK₁⁻¹ and to declare 'panoramic' when the rotation-
  only model (refined from the prior with gates ≈2° → 0.6° → maxErrorPx) explains > 80 % of the inliers; it
  never changes E/F/H/R/t otherwise (tested: identical results with/without prior for general motion).
* **Measured** (synthetic, RADIAL k1 −0.08, N(0, 0.5–1 px), 30–50 % outliers, 6 seeds each): object 0.5 m
  wide/narrow baseline rot ≤ 0.03°, t ≤ 0.08° / 0.37°; scene 50 m wide/narrow rot ≤ 0.04°, t ≤ 0.14° /
  1.3°; backyard 3–100 m t ≤ 0.2°; pure rotation R error ≤ 0.01°; planar pairs within 0.1° of R or
  `planarAlternative`. 2000 matches / 40 % outliers: ~25–45 ms in Node (x86). **Hopeless pairs are
  expensive** (no model ⇒ full budget: ~7 k 5-point + 10 k 7-point samples): ≈ 0.7 s at N = 100–300,
  ≈ 1.1 s at N = 1000 in Node — pass `maxIterations` (≈ 1000–2000) when verifying speculative (retrieval)
  pairs, or pre-filter by match count; N < max(minNumInliers, 8) returns 'degenerate' immediately and the
  budget is capped for inlier ratio max(minInlierRatio, minNumInliers/N).
* **PnP**: Lambda Twist with its own derivation of the plane/cone intersection (the degenerate pencil member
  with the best-conditioned real line pair is used; both lines intersected with the cone a13·M12 − a12·M13,
  scaled by λᵀM23λ = a23, GN-refined); 100 % of 5000 random configurations, ~10 µs/solve warmed. EPnP with
  3 control points for (near-)planar sets (both variants tried when the RMS-thickness ratio < 0.05), β
  candidates from the paper's approximations, each refined by GN on the leading N and on all β; for n ≤ 6
  P3P solutions of the first three points are added (EPnP's linearisation fails for some n = 4 non-planar
  sets). `solvePnPRansac`: P3P + LO-MSAC whose LO step is 4 robust LM iterations (`refine`), then 2 rounds of
  LM (Huber 0.5·threshold) on all inliers with recount. The threshold is maxErrorPx/f in normalised
  coordinates (distortion Jacobian ignored).
* **Triangulation**: DLT is normalised by the mean camera centre and spread (exact to 1e-7 relative at ENU
  offsets of 1e6), rows equally weighted; returns null at infinity or when all centres coincide.
  `refinePoint` is LM with analytic RADIAL Jacobians (single shared camera allowed: `cams.length === 1`).
  `recoverPose`/`cheiralityCount` use midpoint depths and reject points beyond 1000 baselines (COLMAP).
* **Camera**: `undistortNormalized` solves the 1-D radial equation by safeguarded Newton on the monotone
  branch of g(r) = r(1 + k1r² + k2r⁴) (exact to < 1e-10 for |k| < 0.5); beyond the invertible radius of a
  strongly barrel-distorted lens the radius is clamped to the turning point. `scaleIntrinsics` scales f by
  (sx + sy)/2 (COLMAP's rule for single-focal models).
* Scratch buffers are module level: all functions are synchronous and non-reentrant (fine in workers — each
  worker has its own module instance).

### `src/harness/synth` + `e2e/` — review notes (adversarial review; amends the synth notes above)

* **Verified numerically** (`review.test.ts`, `review.gpu.test.ts`), from the consumer side (GT poses +
  meta `R_cam_device` + `imu.json` only): gyro integration reproduces `R_j R_iᵀ = R_cd ΔR_dev R_cdᵀ`
  (< 2e-4 rad), `(accelG − accel)/g` = world up in the device frame (`R_cd` maps it to `R·ẑ`), double
  integration of `R_wd·accelG + g_world` reproduces camera-centre displacements (< 3 mm over 0.75 s), and
  the IMU agrees with the poses at `tTrue`, not at `t`. Rendered frames: markers at the default full-res
  intrinsics land < 0.09 px from their RADIAL projections; frame A warped into frame B through GT depth +
  poses aligns to < 0.015 px (global Lucas–Kanade shift; a 1 px principal-point perturbation is detected).
  ENU ↔ WGS84 uses the exact meridian / prime-vertical radii; heading = optical-axis azimuth,
  `headingDeviceTop` ≈ heading − 90° in the landscape yard.
* **IMU rate is 60 Hz by default** (iOS DeviceMotion is fixed at ~60 Hz, §2.2; was 100 Hz).
  `imu.timeJitterMs` (default 0) jitters the reported timestamps (values stay on the true sample grid).
* **Lawn texture**: the yard ground had ≈ 30–70 SIFT features / Mpx (vs ≈ 1500 elsewhere): blade-only
  detail maps mip-map to flat grey beyond ~2 m. The grass / soil detail maps now carry clumps, tufts and
  stones (1–50 cm), and the detail shader samples each map at two incommensurate scales / rotations so
  the ground never repeats with the tile period. Ground now ≈ 1900–3000 features / Mpx.
* **Metrics**: `scaleError = 1/s − 1` (exact relative scale error of the result, + = too large; was
  1 − s); `denseAlign: 'se3'` uses a rigid Umeyama fit of the centres (it used the Sim3's R, t with the
  scale dropped — off by (s − 1)·R·mean, metres at ENU offsets); unknown align modes / bad CLI numbers
  are errors; the corner distortion error is evaluated on the GT ray through the corner pixel.
* **Generator safety**: `--name` must be a plain directory name (the target is deleted and recreated).

### `src/dense` — fusion & meshing (undistortion, view selection, fusion, TSDF, marching cubes, simplification, point-cloud filters)

* **Files**: `undistort.ts`, `view-selection.ts`, `fusion.ts`, `tsdf.ts`, `marching-cubes.ts`, `simplify.ts`,
  `pointcloud.ts`, `kdtree.ts`, `dense-util.ts` (LRU `DepthMapCache`, `Int3HashMap`, `Yielder`,
  `DenseCancelledError`, local camera helpers), `testutil-scene.ts` (test-only: exact distance to synthetic
  scenes, GT depth stores with noise / outliers). All CPU, no GPU; async stages yield every ~40 ms and throw
  `DenseCancelledError` (`name = 'CancelledError'`) when the CancelToken is set.
* **API superset of §4.6** (documented signatures kept). Extras: `undistortedCamera`, `undistortedSize`,
  `areaResampleRGBA`; selection results also carry `scores`, `neighboursFromSelection(sel, symmetric = true)`
  (best-first neighbour lists for fusion); `FusionOptions` (`minViews` 3, `maxDepthError` 0.01,
  `maxReprojErrorPx` 2 **in depth-map pixels**, `maxNormalErrorDeg` 10, `maxCost`, `step`, `maxNeighbours` 10,
  `cacheSize`, `usePlaneDepth`, `radiusScale`, `maxPoints`, `filteredStore`, `onStats`); `TsdfOptions`
  (`maxBlocks`, `autoVoxelPixels`, `adaptiveTruncationPx`, `sdfMode`, `minWeight`, `maxWeight`,
  `minComponentFraction`, `color`, `maxCost`, `onStats`), `buildTsdfVolume`, `integrateDepthMap`, `TsdfVolume`
  (`fromFunction` for analytic SDFs); `marchingCubes`, `removeSmallComponents`, `meshComponents`,
  `meshEdgeStats`, `compactMesh`; `simplifyMesh(mesh, target, { maxError, boundaryWeight, minNormalCos })`;
  `voxelDownsample`, `statisticalOutlierMask`, `selectPoints`, `filterPointCloud(pc, { voxelSize, statistical:
  { k, std, scaleAdaptive }, minConfidence })`; `KdTree3`.
* **Undistortion**: output camera = `scaleIntrinsics(cam, w, h)` with k1 = k2 = 0 (same f / principal point,
  no crop or FOV change; w, h = round(W·s), round(H·s), s = min(1, maxSize / long side)). Source box-filtered
  (exact area weights) to the output scale, then every output pixel is inverse-mapped through RADIAL and
  sampled bilinearly; the box-filtered image is addressed through the source frame with the per-axis §2 rule
  (rounded sizes give sx ≠ sy). Outside the source or beyond the RADIAL fold radius → RGBA 0.
* **View selection**: score_j = Σ_shared points w_angle(θ)·w_scale(ρ): (θ/3°)² below 3°, 1 up to 45°, Gaussian
  fall-off (σ 15°) above; Goesele scale term. Pairs with 75th-percentile θ < 1° (zero baseline — see the
  PatchMatch review note) or < 5 shared points are dropped; greedy pick with a mild near-duplicate-direction
  penalty; geometric (frustum) fallback up to `minSources`. Depth range = 1st/99th percentile × 0.8 / 1.25 of
  the reference's sparse points (fallback: all points in its frustum; then 0.5–200 × median baseline).
* **Fusion** (COLMAP StereoFusion semantics with two deliberate differences): (1) the depth test uses the
  neighbour pixel's plane (from its normal) intersected with the projection ray instead of the raw pixel
  depth (removes the half-pixel lookup error on slanted surfaces: 87.0 → 85.4 % within 0.5 % without it);
  the reprojection test is forward–backward (neighbour surface point reprojected into the reference);
  (2) traversal depth is 1 (only the reference's neighbours, no BFS through neighbours of neighbours), which
  keeps the resident set to reference + neighbours. Pixels are marked fused only when a point is emitted
  (bit masks per view, 1 bit/px). Point = mean position, normalised mean normal, per-channel median colour,
  radius = 0.7·z/f/√max(cosθ, 0.2), confidence = views / (1 + #neighbours). `filteredStore` receives depth
  maps containing only fused (multi-view consistent) pixels — **use it as the input of `tsdfMesh`**.
  Memory: (maxNeighbours + 2) depth maps resident (20 B/px each) + 1 bit/px/view.
* **TSDF**: `truncation` is in **voxels** (default 4). 8³ blocks in an open-addressing Int32 hash; 11 B/voxel
  (Int16 TSDF normalised by τ, Uint16 weight, 3 × Uint16 colour 8.8, Uint8 colour weight) = 5.6 KB/block;
  default `maxBlocks` 50 000 (≈ 280 MB worst case — lower it on iPhone, e.g. 25–30 k). Auto voxel = 2 pixel
  footprints at the median depth of the DEPTH maps; an allocation pre-pass counts the blocks of the truncation
  bands for a ladder of voxel sizes (×√2 per level, windows of 4 levels per pass) and picks the finest level
  within budget, so a 0.5 m object and a 100 m yard use the same code (×50 scaled sphere test gives identical
  relative error). All blocks exist before integration ⇒ free-space updates of every view carve outliers in
  blocks allocated by other views. Per voxel: point-to-plane SDF (pixel normal; ray distance when cos θ < 0.25
  or no normal) but updates limited to ≤ τ behind the surface ALONG THE RAY (the plane distance alone reaches
  τ/cos θ deep — bulges at edges/thin parts; fixed), linear weight fall-off from −τ/4 to −τ, w = (z̃/d)²·cos θ,
  τ_obs = max(τ, 2·d/f). Blocks culled by frustum and a 16 × 16-px tile max-depth map.
* **Marching cubes**: the 256-case table is generated at load: on every cube face entering/exiting crossings
  are paired walking counter-clockwise (separates the inside corners — depends only on the face's corner signs,
  so adjacent cubes agree ⇒ watertight), loops are triangulated with diagonals only between edges that share
  no cube face (such an edge cannot occur in another cube ⇒ edge-manifold; the noisy-SDF test had 14
  non-manifold edges with plain fans, 0 now; a centre-vertex fallback exists but no case needs it), ≤ 5
  triangles per cube. Vertices welded through an (owner voxel, axis) hash; normals = interpolated central-
  difference TSDF gradients; triangles CCW seen from free space (normals point to the cameras). Edges between
  two saturated values are skipped (ghost surfaces).
* **Simplification**: Garland–Heckbert with area-weighted quadrics, boundary planes (weight 100·|e|²),
  4-ary heap with lazy versioned entries, link condition + boundary-pinch test, normal-flip test
  (cos ≥ 0.2), colours/normals interpolated at the target's projection on the edge. Output keeps the input
  vertex order (compacted).
* **Point-cloud filters — deviation**: SOR uses an exact k-d tree (points stored in tree order) instead of a
  uniform grid; a grid degenerates when spacing varies 100× (mm-spaced object + dm-spaced houses).
  `scaleAdaptive` (default when radii exist) divides each kNN mean distance by the point radius (pixel
  footprint): the plain statistic removed 196/2500 far-wall points and 0/100 outliers in the multi-scale test,
  the adaptive one 0 and 100/100.
* **Measured** (Node 22, x86, one core; tests in `src/dense/*.test.ts`):
  * Undistortion: barrel round trip 0.31 levels mean |Δ| vs a direct pinhole render; dot centroid error
    0.014 / 0.031 / 0.022 px at full / ½ / ⅓ size; 320×240 in 14 ms.
  * Fusion of GT depth maps (σ 0.2 %, 3° normal noise, **10 % outliers**), 16 views 160×120: 99.99 % of points
    within 0.5 % of the true surface, p99 0.26 %, 0 points > 2 %; filtered depth maps 0.001 % bad pixels.
    640×480 × 16 views, 6 neighbours: 1.6–1.9 s (0.33–0.38 µs/px), 746 k points, 100 % within 0.5 %.
    On real `patchMatchCPU` output (photometric + geometric, near scene 256×192, 10 views, 4 sources): raw maps
    → fused 99.53 % within 1 %, 96.9 % within 0.5 % (128×96: 94.9 / 87.0 %, window fattening dominates).
  * TSDF from 32 GT depth maps of a sphere: closed genus-0 mesh, radius error mean 0.024, max 0.14 voxel;
    σ 0.3 % noisy maps: mean 0.041 voxel; yawed box p95 0.48 voxel (corners rounded by < 1 voxel); budget of
    120 blocks → 2 coarsening levels, still closed. Near scene 400×300 × 16 views, 7.5 k blocks: integration
    ≈ 1.3 s (≈ 30 ns per voxel visit), marching cubes ≈ 0.6 s for 708 k triangles.
  * Marching cubes on analytic SDFs: sphere χ = 2, torus χ = 0, max vertex error 0.006 voxel, 0 boundary /
    non-manifold / misoriented edges, also on a high-frequency noisy SDF with 419 components.
  * Simplification to 25 %: two-sided Hausdorff ≤ 0.033 voxel on a 23 k-triangle sphere, colours within 1.1
    levels, no flips, closed; 285 k triangles in ≈ 1.3–1.4 s (≈ 4.7 µs/input triangle, memory-bound).
  * SOR k = 8: 1 M points in 2.3 s (k-d tree build 0.5 s); 93 % of injected outliers removed, 0 inliers.
* **Limitations / TODO**: everything is CPU — a WebGPU TSDF integration kernel (and fusion kernel) would be
  10–50× faster on Apple GPUs; fusion traversal depth 1; marching cubes and simplification are synchronous
  (not cancellable mid-run; run them in the worker); TSDF cannot represent structures thinner than ≈ τ
  (fences, poles) and rounds sharp edges by ≈ 1 voxel; one voxel size per volume (a coarse yard voxel also
  applies to a close object in the same scene — split volumes per region would fix it); SOR uses global
  statistics; depth maps must fit (reference + neighbours) in memory during fusion.

### `src/sensors` — review notes (adversarial review; amends the `src/sensors` notes above)

* **Gyro check D** now decides only after EVERY device axis carried ≥ 2 % of the rotation energy and
  each matched axis correlates ≥ 0.8 (`stats.gyroCheck.excited/minAxisCorrelation`); until then it keeps
  accumulating. Before, a pure pan with the phone upright (rotation about device y only — the usual
  capture motion) "corrected" x/z into a random signed permutation in ~90 % of runs.
* **CompassSensor**: ψ is reset when the relative event stream has a gap > `restartGapMs` (1 s) and on
  `start()` — iOS restarts CoreMotion after a suspension with a NEW arbitrary yaw reference (the old ψ
  biased attitude/heading for ~20 s). `stats.yawResets`.
* **chooseCamera**: several unidentified rear cameras (locale not covered by the label lists; iOS
  enumerates Ultra Wide before Wide) ⇒ the median `focusDistance.min` one (ultra-wide shortest, tele
  longest); `lensMaySwitch` is set for such an ambiguous Apple choice, and an unidentified Apple rear
  camera gets the main-camera FOV prior (not the 63° webcam prior).
* **CameraCapture**: `restart()` is superseded by a concurrent `stop()`/`start()` (its stream is stopped;
  no leaked live camera, no 'running'/'error' state after stop); only automatic restarts count towards
  the 3-per-10 s limit ('manual'/'visible' do not); the `<video>` is `play()`ed again on unmute/visible
  (WebKit pauses media of a backgrounded page ⇒ rVFC stalls); returning to visible with a live,
  unmuted track starts the new `segment` immediately (frames no longer carry the old segment for 1.2 s).
* **FramePump**: a grab still in flight at `stop()` is discarded (bitmap closed, IMU not drained); failed
  grabs count towards the rate limit (no 30 Hz error storm).
* **WakeLock**: a late `release` event of a previous sentinel (iOS dispatches it after resume) no longer
  orphans the re-acquired lock. **GeoSensor.approximate** follows the newest fix (not sticky after a
  coarse first cell/Wi-Fi fix).
* Tests: `src/sensors/review.test.ts` (Node) and `src/sensors/review.gpu.test.ts` (Chromium).

### `src/geometry` — review notes (adversarial review; supersedes conflicting points of the `src/geometry` notes above)

* **recoverPose tie-breaking**: when several of the four (R, ±t) candidates have the same finite-depth
  in-front count — in particular 0 for all of them when every point is beyond `CHEIRALITY_MAX_DEPTH` (1000)
  baselines or the rays are exactly parallel (pure rotation) — the candidate was the first one, i.e. the
  180°-twisted rotation half of the time. Ties are now broken by the in-front count without the depth
  bound, then by Σ cos∠(R x1, x2) (rejects the twisted R). `numInFront` keeps its finite-depth meaning.
  `cheiralityCount(..., maxDepth = Infinity)` is supported.
* **essential5pt**: real eigenvalues whose Gauss–Newton polish does not converge (rel. residual > 1e-6;
  ill-conditioned root clusters near pure rotation) are dropped — they were returned although they are
  not essential (σ2/σ1 down to 0.96). True solutions polish to ≤ 1e-9 (4800 noise-free configurations
  incl. planar / forward motion / baseline/depth 1e-4). The polish now reports the residual of the
  accepted point, not of a rejected trial step. Still lost at small baselines: ~2–10 % of noise-free
  samples at baseline/depth 1e-3…1e-2 (up to 30 % for forward motion toward a plane), 25–80 % at
  1e-4…1e-3 — the true root's eigenvalue is perturbed into a complex pair by the ill-conditioned
  elimination template (widening the near-real gate does not help); such parallax is below the noise.
* **estimateTwoViewGeometry**:
  - planar_or_panoramic: the H decomposition is computed once; if the E-RANSAC pose matches an H branch
    with clearly less cheirality support than the best branch (> max(2, 1 % of nH) fewer points in front),
    the best branch is used — E cannot tell the planar branches apart, cheirality often can (walking along
    a facade / sideways over the ground: wrong branch, 5.7°–19° rotation error, in 3–5 of 6 seeds before).
    'planar' reports, among branches tied in support, the one closest to the selected pose;
    `planarAlternative` only if another branch has ≥ 90 % of the best support (genuinely ambiguous).
  - 'panoramic' with an observed translation (≥ 80 % of the inliers in front for the full model's pose;
    pure rotation measures ≤ 0.52) reports the full model's R; the rotation-only fit is used only for
    (near-)pure rotation. It absorbed the translation: 0.11–0.28° error vs 0.006–0.04° now (3-D scene
    5–15 m, 5 cm step). t stays (0, 0, 0).
  - Inlier sets that are coincident / collinear within maxErrorPx (minor-axis RMS of the 2-D scatter in
    either image) or have < minNumInliers distinct correspondences (1-px cells, min over both images) are
    'degenerate' (were 'calibrated' with an arbitrary pose for identical, collinear or 3-point-repeated
    input).
* `refinePoint` with `cams = []` / `n = 0` returns a copy instead of throwing.
* Regression tests: `src/geometry/adversarial.test.ts` (26 tests; the 8 marked [regression] fail on the
  pre-review code). Remaining known limits: planar pairs with t ∥ n (walking toward a wall) are
  ill-conditioned for both H and E (≈ 0.2–0.3° / 2–6° from two views); hopeless (random) pairs still cost
  ≈ 0.6–1.2 s in Node (pass `maxIterations` for speculative pairs); `projectCam` does not reject points
  beyond the monotone radius of a strongly barrel-distorted lens (they fold back into the image).

### `src/features` — GPU review notes (adversarial review; amends the `src/features` GPU notes above)

* **Kernels verified line by line against the CPU reference.** Checked: barriers and
  shared-memory reuse in all 13 WGSL kernels, uniformity, uniform layouts against `ParamBlock`,
  index bounds and the fixed-point overflow bound. No kernel defect found. New parity tests
  show identical feature counts, position agreement 1.000 and median descriptor cosine 1.00000
  for each of these: S = 1 / 2 / 4, `maxOrientations` 4, σ below the base blur (copy path),
  `firstOctave` 1, `distribute: false`, odd sizes, a 700×40 image. Tiled and direct blurs are
  bit-identical at odd sizes and with radius 45 (above the tiled-V limit). A submit budget of
  63 command buffers gives bit-identical output. Multi-pass matching (`rowTile0 > 0`), extreme
  column chunking, exact ties, all-zero descriptors and `ratio > 1` give pairs identical to
  matchCPU.
* **Fixed: GPU errors were misclassified as dropped work.** An invalid command buffer is not
  executed, so the work stamp also mismatches, and the error scope's validation / OOM error
  was ignored in favour of that mismatch. Result: 3 retries, then a *transient*
  `GPUWorkDroppedError`, so the pipeline kept a broken GPU backend for the whole session (3
  failing GPU attempts per call, then the CPU). A scope error now outranks the stamp. Such a
  failure is a hard `GPUFeatureError` and drops the cached per-size resources.
* **Fixed: input errors no longer switch the pipeline to the CPU for good.** These are now
  plain `Error`s (`FeatureInputError`, exported) instead of GPU failures:
  * FeatureSets shorter than `count` (before: a RangeError → `GPUFeatureError`, or an
    out-of-view read);
  * unreadable `extractFromBitmap` sources;
  * a detached `ImageData` (Chromium's `copyExternalImageToTexture` crashes the renderer on
    it);
  * non-Uint8 / Float32 gray data (before: uploaded as raw bytes).
* **Fixed: freeing resources mid-call.** `releaseCaches()` / `clearCache()` / `destroy()`
  during a running call are deferred to its end. Before, they destroyed buffers still used by
  later command buffers, so the call silently retried and re-allocated what was meant to be
  freed.
* **Fixed: a transient start-up self-test failure** (GPU work dropped while the app was
  backgrounded during `createFeaturePipeline`) is postponed to the next call instead of
  selecting the CPU for the session. `selfTest()` results have `transient`. The SIFT
  self-test no longer fails for `maxFeatures < 10`.
* **Fixed: smaller issues.**
  * `extractFromBitmap(bmp, fullWidth)` with only one dimension: the CPU fallback used an
    anisotropic frame.
  * Error scopes in `compilePipeline` are synchronous, and the two pops in `errorScope` are
    issued together: concurrent compilations no longer swap errors. Before, a good pipeline
    could report another pipeline's shader error.
  * `BudgetEncoder` creates encoders lazily, so no stamp-only encoder is left dangling after
    the last submit.
  * Octave sizes are checked before any allocation (fixes a leak on the limit error).
  * Staging buffers are always unmapped.
* **Still open:**
  * `SiftGPU` / `MatcherGPU` hold their error scopes across `await`. A GPU error from another
    module during that window, if that module uses no scopes of its own, is attributed to
    features, which then fall back to the CPU. The reverse attribution cannot give wrong
    results: an invalid command buffer fails the stamp check.
  * Float32 input must stay within 0..1. The descriptor fixed point has about 25 % headroom;
    0..255 floats overflow the u32 bins silently.
  * `onSubmittedWorkDone` has no timeout. If iOS stalls the queue while hidden, the call
    waits until the page is visible again.
* Tests: `gpu-review.gpu.test.ts` (23, Chromium) and `gpu-review.test.ts` (8, Node, with a
  fake device for `BudgetEncoder` / `WorkStamp`). The existing GPU tests now require identical
  match arrays, and stricter SIFT thresholds: position ≥ 0.97 and cosine ≥ 0.995 (before
  0.85 / 0.97).

### `src/dense` — fusion & meshing review notes (adversarial review; amends the fusion & meshing subsection above)

* **TSDF allocation** (`allocateBands`): one stride from the 2nd depth percentile left holes.
  Pixels more than about 3× that depth were spaced over a block apart, and grazing surfaces too.
  Measured on a near box plus a wall 15× further: 1 223 of 1 319 wall pixels and 285 box pixels
  had no block. Sampling now adapts per pixel: a power-of-two lattice stride
  p = ½·block·cos θ / footprint, and m × m sub-pixel rays (m ≤ 6) where a pixel is wider than
  half a block. The last row and column are always sampled. All ladder levels share one pass
  over the pixels. Result: 0 missing blocks, and the far wall is meshed with 100 % coverage
  (was 57 %).
* **Grazing surfaces** (a ground plane seen from 1.6 m): the yard mesh had 1 128 components,
  because no voxel below the ground was ever observed. Behind the surface, updates stopped at
  τ_obs along the ray, and below cos θ = 0.25 the ray distance was used, which saturates the
  voxel above the ground to +1. Both pushed the cubes into ghost rejection or left them with no
  sign change. Now:
  * the plane distance is used down to cos θ ≥ 0.1;
  * the reach behind the surface along the ray is max(τ_obs, 1.5·voxel / cos θ), at most
    10·τ_obs;
  * updates beyond τ_obs along the ray carry 0.1× weight, so voxels seen directly from another
    view dominate;
  * the allocation band and the tile culling use the same reach.

  Effect on the yard: largest component 50 k → 157 k triangles, all components 171 k → 293 k
  triangles. On a pure ground plane, the share of ground within 25 m covered by the largest
  component rose from 73 % to 96.5 %. Sphere accuracy improved: mean 0.024 → 0.015 voxel, max
  0.14 → 0.095. Trade-off: on the yawed box, p95 goes from 0.48 to 0.54 voxel because box edges
  seen at grazing angles get slightly rounder.
* **Tile culling**: the projected block radius f·r/(Z − r) under-estimates an off-axis sphere,
  by up to 2× at 120° FOV, so blocks that were in view got culled. Now f·r·√(1 + (X/Z)²)/(Z − r)
  per axis. The voxel update count matches a brute-force integrator exactly, in ray and plane
  mode.
* **Memory cap**: with a user-given `voxelSize`, the counting hash map was unbounded (cap = ∞).
  It is now capped at `maxBlocks`, followed by greedy allocation that stops at the budget.
  The module-level per-pixel scratch (16 B/px) is released after `buildTsdfVolume`.
* **Non-finite input**: one `Infinity` depth made the band sample count infinite. NaN block keys
  all hash to one slot, so probing went quadratic and the run hung. NaN and ±∞ depths are now
  invalid everywhere: TSDF, fusion, and the scene-scale sampling. In fusion, an `Infinity`
  neighbour depth also passed both consistency tests (`Inf > Inf` is false), and NaN normals
  leaked into the output. Both are fixed.
* **Small components**: `minComponentFraction` (2 % of the largest) also deleted real structure
  in large scenes, e.g. a 7 m tree at 40 m (3.1 k triangles, 1.99 %). New
  `componentKeepVoxels` (default 16): components whose bounding-box diagonal exceeds this many
  voxels are always kept. `removeSmallComponents(mesh, frac, minTriangles, keepDiagonal)` is
  backwards compatible.
* **SOR**:
  * In scale-adaptive mode, points with radius 0 got an infinite statistic and were all
    removed. They now use the median radius.
  * The variance used Σv²/m − μ² and the threshold had no rounding slack, so a regular cloud with
    identical statistics was deleted completely (4 000 of 4 000 points). Now a two-pass variance
    with a 1e-9 relative tolerance.
* **View selection**:
  * The frustum fallback for fewer than 10 observed points replaced the observed points instead
    of adding to them. Observed points outside the pinhole frame (distorted cameras) were lost,
    falling back to a range of 0.5–200 × baseline. It now uses the union.
  * `minSources` could exceed `numSources`; it is now clamped.
* **Verified correct (adversarial tests in `fusion-adversarial`, `meshing-adversarial` and
  `tsdf-adversarial` `.test.ts`)**:
  * undistortion direction and §2 scaling: image smaller than the calibration, anisotropic
    rounding with and without antialiasing, error ≤ 0.02 px;
  * fusion: world-frame normals with rotated cameras, z-depth at 120° FOV, cancel mid-run;
  * marching cubes on a random field over 3³ blocks: all 256 cases occur; closed, edge-manifold
    and consistently oriented, with positive enclosed volume; exact zero voxels give χ = 2;
  * `EdgeHeap` (exported for tests) against a reference under 60 k random pushes and pops;
  * simplification of a flat grid stays planar, keeps its outline and area, and has no flips;
    a sphere at 1 % stays closed with χ = 2; degenerate and non-manifold input does not crash;
  * the k-d tree on duplicate and collinear points;
  * TSDF face winding, and cancel mid-integration.
* **Still open**:
  * marching cubes and per-view integration are synchronous; cancel latency is one view's
    integration, ≤ ~1 s at 50 k blocks;
  * one voxel size per volume, from the median depth: for an object in front of a room, the
    room decides. The engine should pass `maxDepth`, e.g. 2 × the object distance, in object
    mode;
  * grazing ground with cos θ < 0.1 (under about 6°, beyond ≈ 10× the camera height) uses the ray distance and can still fragment where few views overlap.

### `src/sfm` extras (keyframes, pairs, vocab, inertial, geo, world)

* **Files**: `keyframes.ts` (KeyframeSelector, rotationCompensatedParallax), `pairs.ts` (sequential /
  retrieval / spatial pairs, mergePairs), `vocab.ts` (k-means vocabulary + TF-IDF index), `inertial.ts`
  (time offset, gravity, metric scale, gravity alignment), `geo.ts` (WGS84/ECEF/ENU, GPS alignment, compass
  yaw), `world.ts` (computeWorldAlignment). `extras-testutil.ts` is TEST-ONLY (continuous orbit / walk
  trajectories, 100 Hz IMU, keyframes in a random similarity frame, GPS). No imports from the SfM core.
* **Time offset sign**: an image stamped `t_cam` was exposed at IMU time `t_cam + offsetMs`.
* **KeyframeSelector**: blur = sharpness < 0.6 × running median of the last 30 frames (rejected frames
  included so the reference adapts); the FIRST keyframe waits for 5 frames of statistics. Overlap =
  inliers / best inlier count observed since the last keyframe (independent of the absolute match rate).
  Parallax trigger 3 / 4 / 5 % of the width (object / auto / scene), rotation 12°, tracking lost < 30
  inliers, min 150 ms, max 2 s only while moving; soft triggers wait ≤ 250 ms for a frame ≥ 0.85 × median
  sharpness. `evaluate` commits 'keyframe' decisions itself; `revertLastKeyframe()` if the engine drops it.
  `rotationCompensatedParallax` undistorts with `CameraIntrinsics` (RADIAL), or applies K R K⁻¹ to raw pixels
  with a Mat3 K.
* **VocabularyIndex**: exact nearest word (argmax x·c − ½‖c‖²). Uses `features/simd` `createSimdKernels().dot4`
  (int16 SIMD dot products, centroids rounded to 1/64 grey level) and re-scores in float64 every centroid
  within the rounding bound of the best ⇒ identical to the JS search (tested), 4000 features × 512 words
  ≈ 30 ms (JS fallback ≈ 300 ms). Training: k-means++ on 8·k descriptors, ≤ 6 Lloyd passes on 16·k
  (SIMD assignment; JS fallback with Hamerly bounds + exact handling of the 32 largest centre moves):
  k = 512 ≈ 1 s (JS 3.3 s). IDF is recomputed lazily after add/remove; query = forward scan over the
  sparse BoWs (500 images ≈ 0.5 ms). `exportCodebook()` / `new VocabularyIndex(codebook)` for persistence.
* **Inertial**: R_wd(t) = vision keyframe attitudes + gyro (4th-order Magnus) with the per-interval
  mismatch distributed linearly (`AttitudeInterpolator`). Time offset: robust (Cauchy) misfit of
  log(V_kᵀ G_k(δ)) with the gyro bias re-estimated per δ (Forster et al. bias Jacobian), 2 ms grid over
  ±250 ms + golden section; σ from the LS curvature. Gravity: weighted mean of R_wd·accelG over ALL samples
  in the keyframe span (fallback: keyframe `gravityCam`). Scale: the SAME linear operator (smoothing-spline
  second derivative through the keyframe samples → 20 Hz grid → zero-phase 0.3–3 Hz Butterworth) is applied
  to the camera centres and to double-integrated world-frame IMU acceleration sampled at the keyframe times,
  so keyframe-rate aliasing affects both sides identically; IRLS regression vision ≈ s₀·IMU + world bias +
  device-frame bias (Gaussian prior 0.05 m/s² — essential for orbits, where a device bias along the optical
  axis mimics the centripetal acceleration). σ = Newey–West HAC × 1.3 (calibrated on 30 seeds each).
* **Measured** (30 seeds each; gyro noise 0.005 rad/s, bias ~0.01; accel noise 0.05, bias ~0.05 m/s²;
  keyframes 2–4 Hz in a random Sim3 frame, offsets ±150 ms): orbit 0.4 m @ 0.2 Hz (hand-held: ±15 % speed,
  ±3 cm radius, 3–6 mm jitter) scale rms 0.5 % / max 1.0 %; 40 m walk @ 1.2 m/s with 2 Hz bob rms 1.0 % /
  max 2.4 %; z-score rms 0.83–0.94; time offset max error 0.9 ms; gravity rms 0.22° (orbit) / 0.44° (walk,
  max 0.98° — an uncompensated horizontal accel bias tilts gravity by b/g when the device hardly rotates).
  Total `estimateInertialAlignment` ≈ 20–30 ms (Node). Smooth constant-rate pans are flagged unreliable.
* **GPS**: fixes associated by time (centre interpolated between bracketing keyframes), weights 1/σ².
  With gravity (+Z up) a 4-DOF fit (complex regression EN ≈ a·XY + b, 2-point RANSAC + Huber IRLS; vertical
  from altitudes or mean camera height 0) — works for straight tracks; else Sim3 (`alignSim3Ransac` +
  weighted Umeyama). Requires ≥ 5 fixes and a noise-corrected track extent √(12·(λ_max − σ̄²)) > 4 × median
  accuracy. σ_s inflated by (1+ρ)/(1−ρ) for lag-1 residual autocorrelation. 40 m walk, 3 m noise: scale
  error ≤ 4.8 % (12 seeds), max |z| 1.5.
* **computeWorldAlignment**: gravity (IMU, else `gravityCam`) → R_g; GPS 4-DOF yaw, else compass yaw
  (`compassReference` 'auto' = device top while within 45° of horizontal, else optical axis — the
  `CompassSensor` convention; heading must be TRUE north); scales fused by inverse variance in log-space,
  or the more certain one when they disagree by > 3σ (warning); GPS ⇒ ENU translation +
  `originLatLonAlt` (alt = mean MSL altitude, 0 without altitudes), else first camera at the origin.
  Apply with `Reconstruction.transform(s, R, t)`; `transformPose` gives R' = R_p Rᵀ, t' = s t_p − R' t.
* **Limits**: `estimateTimeOffset` cost ∝ keyframes × 250 grid points (2000 keyframes ≈ 1 s); a single
  constant gyro bias and constant accelerometer biases are assumed per capture; the scale needs
  band-limited (0.3–3 Hz) acceleration ≥ 0.08 m/s² RMS (`excitationRms`) — tripod-smooth motion gets
  `reliable: false`; segments are split at keyframe gaps > 3 s / IMU gaps > 250 ms.

### `src/sfm` core (database, correspondence graph, reconstruction, triangulator, mapper, live, synthetic)

* **API superset of §4.5** (documented signatures kept). Extras: `pairIds`, `swapMatches`, `invertTwoView`,
  `MAX_IMAGE_ID` (2^20), `FeatureDatabase.colors` (3 B/feature, optional 3rd arg of `addFrame`), `removeFrame`,
  `deletePair`, `numPairs`. `getTwoView(a, b)` with a > b returns the cached inverse (inliers swapped,
  x_a = Rᵀx_b − Rᵀt, Eᵀ, Fᵀ, H⁻¹).
* **CorrespondenceGraph**: one singly linked list per (image, feature) in a shared typed-array pool (12 B per
  directed correspondence, no objects). Hot loops: `heads(img)` + `nextOf/imageOf/featureOf` (re-read after
  `addTwoView`, the pool grows); `getCorrespondences(img, f, outImg, outFeat)`; `correspondences()` allocates
  (convenience only). `addTwoView` of an existing pair REPLACES it; `removeTwoView`, `matchesBetween(a, b)`
  (oriented), `neighbours(img)` (adjacency map), `numObservations`, `isTwoViewObservation`, `version`.
* **Reconstruction**: `addImage(id, cameraId, numFeatures, keypoints?)` (keypoints stride 4 are needed for
  errors / filtering / visibility), `addCamera(cam, id?)` (copied). `RecImage` extras: `keypoints`,
  `numVisiblePoints3D`, `visibilityScore` (COLMAP 6-level visibility pyramid, maintained incrementally once a
  graph is attached with `attachGraph`; `syncVisibility(id)` recomputes one image after the graph grew).
  COLMAP semantics: deleting an observation of a 2-track deletes the point; `mergePoints` = track-length
  weighted position/colour, concatenated track, both ids deleted. `transform(s, R, t)`: X' = sRX + t,
  R'ᵢ = RᵢRᵀ, t'ᵢ = s tᵢ − R'ᵢ t. `meanReprojectionError()` = mean over ALL observations, recomputed (not
  the cached `error`); `updatePointErrors()`. Filtering: `filterPoints3D / filterPoints3DInImages /
  filterAllPoints3D(maxPx, minTriAngleRad)`, `filterObservationsWithNegativeDepth`, `filterImages`;
  `hasBogusParams`, `reprojectionErrorSq`, `imageLUT()`, `numObservations`, `numSharedPoints`. `clone()`
  shares keypoint arrays and the attached graph.
* **Triangulator** (COLMAP defaults: create/continue 2° angular error, merge/complete 4 px, completion 5 hops,
  min angle 1.5°, re-triangulation of pairs < 20 % triangulated with 5°, ignore two-view tracks).
  `RayTriangulator`: 2-view minimal samples (exhaustive for ≤ 15 observations), angular residuals, cheirality
  via the camera z-axis, LO = weighted LS ray intersection (re-weighted by 1/distance² ≈ angular error; no
  DLT/SVD, no allocation). Deviations: a track never gets two observations of the same image
  (continue/complete skip images already in the track); `completeImage` only scans the image's own
  correspondences (COLMAP re-scans every element of its tracks — 5× slower; `completeAllTracks` after global
  BA covers the rest); `completeAllTracks` iterates untriangulated features (fewer) instead of all tracks;
  `mergeAllTracks` scans each graph edge once.
* **Mapper** (`MapperOptions`, `DEFAULT_MAPPER_OPTIONS`, `resolveMapperOptions`): cameras are shared per image
  size (or `cameraKey(frameId)`), initial guess from `intrinsics` (object or per-frame function, rescaled to the
  frame size) else `hfovPriorDeg` = 69° across the long side. Initial pair: first image by #correspondences,
  second image by (stored two-view triAngle ≥ threshold) then #correspondences — i.e. video pairs further apart
  are tried first; pairs whose stored angle is < ½ threshold or 'panoramic' are skipped without
  re-estimation; re-estimation with current intrinsics (≤ 100 per search, cached); 'uncalibrated' pairs get
  their pose from E = K₂ᵀFK₁; relaxation (inliers, angle): (100,16°) → (50,16°) → (50,8°) → (30,8°) →
  (30,4°), also for later models. Initial points need ≥ max(1.5°, ¼·init angle). Registration: PnP 12 px,
  ≥ 30 inliers AND inlier ratio ≥ 0.25 (acceptance, not only the RANSAC cap). Local BA: image + 6 neighbours
  (COLMAP's 6 includes the image), soft-L1 1 px, ≤ 25 iterations, ftol 1e-4, **observations of constant
  (outside-anchored) points subsampled to ≤ 600 per image** (2.5× faster, accuracy unchanged in tests); after
  it, only the observations IN the bundle images plus the modified points are filtered (COLMAP re-checks every
  observation of every point of the bundle images — with 60-view tracks that is the whole model). Global
  BA: trivial loss (as COLMAP), f + k1 + k2 refined once ≥ 3 images are registered, gauge = first registered
  image fixed + the largest relative-translation component of the second. **Intermediate** global
  refinements (10 % growth) use ≤ 12 observations per track (every ⌈L/12⌉-th; the Schur cost is Σ m²) and ≤ 2
  BA/filter rounds; the **final** one uses all observations and ≤ 5 rounds. Above 150 images with mean track
  > 8 the global BA uses PCG (dense reduced camera matrix: 1.5–2× faster per LM iteration than the skyline
  Cholesky at 160–250 images, same iteration count). Images are filtered (bogus camera or < 6 points) once
  ≥ 20 are registered. `runIncrementalSfM` yields every ~25 ms between steps, runs global BA in 500 ms slices
  (LM damping carried over: identical results) and the initial-pair search in chunks of 6 re-estimations;
  longest synchronous step ≈ 0.6–0.8 s (desktop Node, idle machine, 100 images). Cancel ⇒
  `SfMCancelledError` (name 'CancelledError'). Models smaller than min(10, 0.8·N) images are dropped unless
  it is the only one. Extras: `buildCorrespondenceGraph`, `extractColors`, `intrinsicsFromFov`,
  `createMapperSharedState`, `IncrementalMapper.timings`.
* **LiveMapper** (`LiveMapperOptions` = mapper options + live ones): the engine must add the keyframe to the DB
  and its verified pairs to the graph BEFORE `addKeyframe`. Initialises from the best pair among the last 30
  keyframes (≥ 60 inliers, ≥ 5° median angle, ≤ 6 re-estimations per call), then registers the pending
  keyframes; each new keyframe: PnP → triangulate → local BA (120 ms BA budget); failures stay pending and are
  retried (nearest first, ≤ 4 per success) when their visible-point count grew (relocalisation). Global BA
  with an 800 ms budget at 3, 4, 6, 8, 11, … ≤ 30 images, then every 15 registrations. `refine(ms?)` for a
  final global refinement; `incrementalMapper`, `pendingIds()`, `timings`. No re-initialisation of a second
  model while lost (offline `runIncrementalSfM` handles that).
* **synthetic.ts** (test data for any module; not re-exported): `generateScene({ kind: 'orbit' | 'walk', … })`
  (orbit: object 0.3 m at 0.5 m; walk: 60 m yard, houses/trees 5–50 m, sideways walk with panning), RADIAL
  projections + noise, clutter features, RootSIFT-like descriptors (true-match distance ≈ 0.4, random ≈ 0.75),
  putative matches with exact outlier ratio, `buildSyntheticDatabase` (two-view verification with a given
  camera guess, or GT inliers with `verify: false`), `combineScenes` (disconnected captures),
  `evaluateReconstruction` (Sim3 ATE, rotation, focal, k1/k2, wrong-observation rate). `e2e-testutil.ts`:
  acceptance-setting data + keyframe streaming like the engine; `rotationOnlyDb()`.
* **Measured** (Node 22 x86, single process, 0.7 px noise, 25 % outlier matches, f +8 %, k1 = 0.03):
  orbit 60 images — 60/60, ATE 2.7e-5 of the extent, focal error 2e-6, k1 0.0301, mean reprojection 0.84 px,
  ≈ 10–12 s; walk 100 images — 100/100, ATE 4–5e-5 of the 81 m extent (4 mm), focal 1e-4, 0.85 px,
  ≈ 12–14 s. LiveMapper: 60/60 and 100/100, same accuracy, per keyframe median 90–130 ms, p90 250 ms,
  max ≈ 0.7–1.1 s (global BA keyframes). Disconnected orbit + walk → 2 models. Stress (40 % outliers + f
  +15 %, or f −10 % with k1 = ±0.05/0.06): all registered, ATE ≤ 5e-5, focal ≤ 2e-4. Scaling: walk with
  250 keyframes and 58-view mean tracks (358k observations) ≈ 150 s, ATE 7e-5, focal 4e-6 — global BA
  dominates (≈ 1–3 s per LM iteration at the end); the keyframe selector should keep tracks moderate.
* **Limits / TODO**: no intrinsics prior in BA (early self-calibration with few images relies on the ≥ 3-image
  rule); `planarAlternative` of planar initial pairs is not disambiguated; principal point never refined by
  default; no loop-closure-specific handling beyond the correspondence graph (retrieval pairs must be in it);
  live mode keeps a single model.

### `src/sfm` extras — review notes (adversarial review; amends the `src/sfm` extras notes above)

* **Time offset from linear acceleration** (new `estimateTimeOffsetFromAcceleration`): when the device hardly
  rotates (steady camera while walking / sliding) the gyro-based offset is unobservable and the default (0) was
  used; a true 40–150 ms offset then biased the IMU scale by 4–9 % while it was still reported `reliable`
  (σ ≈ 3 %). The fallback minimises 1 − R² of the scale regression over the offset (25 ms grid ±250 ms +
  golden section, offsets with a non-positive scale rejected; σ from the curvature scaled by the HAC effective
  row count; ≈ 35 scale fits ≈ 0.2 s Node). `estimateInertialAlignment` and `computeWorldAlignment` use it only
  when the gyro offset is unreliable; results report `offsetSource: 'gyro' | 'accel' | 'default'` and
  `accelTimeOffset`. Steady-camera walks/orbits: offset error < 1 ms, scale error < 0.7 % (was 4–9 %).
* **Robustness**: `GyroIntegrator` drops samples with a non-finite gyro (one NaN made every later attitude NaN);
  `estimateMetricScale` ignores non-finite accelerations. `alignToGps` ignores fixes with a non-finite accuracy
  (one such fix made the origin NaN ⇒ null alignment) and falls back to 2 × σ_h for a non-finite
  `altitudeAccuracy` (was NaN translation); `yawFromCompass` treats a non-finite accuracy as 20°.
  `VocabularyIndex.addBow` / `query(BowVector)` drop word ids ≥ k and non-positive / non-finite tf (were NaN
  scores for the whole document). `spatialPairs` never pairs cameras with a non-finite pose; 180° accepts all.
* **KeyframeSelector**: `sharpnessWarmup > sharpnessWindow` deadlocked (no first keyframe ever) — warm-up is now
  clamped to the ring size; `revertLastKeyframe` keeps a 16-deep history (two reverts used to drop the reference
  entirely ⇒ the next frame was a spurious 'first' keyframe while `numKeyframes` > 0); a non-finite `t` is
  skipped (could commit a NaN keyframe time); NaN `numInliersToLastKF` = "not measured" (no lost / overlap
  trigger). `rotationCompensatedParallax` skips NaN distances (non-finite / out-of-range keypoints).
* **mergePairs** (also behind `sequentialPairs` / `retrievalPairs` / `spatialPairs`): typed-array sort of numeric
  keys for integer ids in [0, 2²⁶) — 400 k pairs ≈ 0.1 s (was ≈ 1.5 s); other ids (≥ 2²⁶, negative) use a
  collision-free generic path (pairId would collide); non-finite ids dropped.
* Verified correct (no change): R_wd = R_wc·R_cam_device, gyro right-multiplication and the Magnus sign
  (h²/12 ωa×ωb), Forster bias Jacobian sign, offset sign (τ = t_cam + offset), specific-force gravity sign,
  Reinsch spline bands / banded Cholesky, RBJ Butterworth + DF2T steady-state init, filtfilt odd padding, scale
  model signs (bias_w = −c_w/s₀, bias_d = −c_d/s₀), WGS84/ENU formulas, 4-DOF complex regression and σ,
  compass yaw sign, `transformPose` vs `Reconstruction.transform`, SIMD rounding bound in `Quantizer`
  (E = (‖x‖₁ + max‖c‖₁)/2 + ¼ in 1/64 units). SIMD and JS k-means give identical codebooks (tested).
  Measured z-scores (30 seeds, 60 Hz IMU, ±2 ms jitter): time offset rms 0.96–1.12; gravity err/σ rms 0.4–1.1.
* Tests: `src/sfm/extras-review.test.ts` (22 tests; 12 reproduce the defects above on the pre-review code, the
  rest are adversarial guards: π rotations, antimeridian / poles, 1e±3 units, 60 Hz jittery IMU, search-range edges).
* **Remaining**: yaw comes from GPS (4-DOF) whenever the GPS alignment succeeds, even when its `yawStdRad` is
  larger than the compass σ (short backyard tracks) — no fusion; `estimateGravity`'s `gravityCam` fallback σ
  ignores temporal correlation (optimistic); errors-in-variables attenuation of the IMU scale (≈ σ²_noise /
  var(F[P]), ≈ 2 % only at the 0.08 m/s² excitation floor with the pessimistic simulated 0.05 m/s² noise) is not
  corrected; TF-IDF with very few indexed images is degenerate (N = 2 ⇒ shared words have idf 0).

### `src/app` + `src/workers/protocol.ts` (mobile UI, engine client, protocol)

* **Protocol** (`src/workers/protocol.ts`, ENGINE.md §2–3 types verbatim + extensions, nothing
  removed): `EngineMethodMap` types every request (`init`, `frame`, `finishCapture`, `refine`,
  `densify`, `mesh`, `export`, `loadSession`, `cancel`, `dispose`) plus **`visibility`**
  `{hidden}` (workers get no `visibilitychange`; the engine must not submit GPU work while hidden).
  Requests `{id, method, params}`; responses `{id, ok, result | error}`; events `{type: 'progress'
  (requestId?, stage, progress, message?) | 'live' (update) | 'log' | 'error' (fatal, reason)}`.
  Errors cross as `SerializedError {name, message, stack?, code?}`; well-known names in
  `ENGINE_ERRORS` (`CancelledError`, `NotImplementedError`, `EngineCrashedError`,
  `EngineTimeoutError`, `EngineBusyError`, …). Extensions: `EngineOptions.resume` (re-open the
  persisted `sessionId` after a worker respawn), optional FramePump metadata on `FrameInput`
  (`index`, `segment`, `gravityDevice`, `attitude`, `imuCalibrationVersion`, `timeSource`),
  `loadSession(id, options?)` may be called without `init` and returns `SessionSummary` WITH the
  stored models (`sparse`, `dense`, `mesh`). `LiveUpdate.overlay` is interleaved (u, v) in [0,1]
  relative to the frame AS SUBMITTED (un-rotate if the engine works in a canonical orientation).
  `live` events are only for updates NOT returned as a frame response. A worker that cannot
  continue (GPU process gone: `requestAdapter()` null) posts `{type:'error', fatal:true,
  reason:'gpu-lost'}`. `collectTransferables(obj, {minByteLength = 1024})` walks arrays /
  plain objects / Maps / Sets: ArrayBuffers of typed arrays (each once; never SAB or detached),
  ImageBitmap, OffscreenCanvas, VideoFrame, MessagePort. Small arrays (poses) are cloned so the
  sender's copies stay valid — only pass objects whose large buffers the sender gives up.
* **EngineClient** (`engine-client.ts`, implements `EngineLike`): one frame in flight
  (`EngineBusyError` otherwise, bitmap closed), `processFrame` resolves with the LiveUpdate and
  also emits it to `onLive`; timeouts: init 120 s, frame 30 s, long operations = 10 min of
  INACTIVITY (reset by progress). Respawn (terminate + new worker + replay last `init` with
  `resume: true` or last `loadSession`; pending requests reject with `EngineCrashedError`) on:
  module load failure, fatal error event, init timeout, 2 consecutive frame timeouts. Cap 3 per
  60 s → state `dead`; after 2 GPU-related failures the replay uses `preferGPU: false`.
  `src/workers/engine.worker.ts` is a PLACEHOLDER (every request → `NotImplementedError` "engine
  not implemented yet"); the app shows that clearly and suggests `?mock=1`.
* **MockEngine** (`mock-engine.ts`, `mock-scene.ts`; `?mock=1`, `mockSpeed=<x>`,
  `mockFail=<method>[:msg]`): procedural object-on-table / backyard scenes; keyframes every 10° of
  an orbit driven by frame timestamps; overlays are projections of the scene into the moving
  camera; refine / densify / mesh are timed stages; outputs are real `PointCloud` /
  `TriangleMesh`, exports go through `src/io` (PLY, GLB, COLMAP ZIP with placeholder JPEGs).
* **App** (`app.ts`): pure navigation state machine in `router.ts` (no History API — single
  document, see iOS notes), `Store` (`store.ts`), screens in `screens/*`, components in
  `components/*` (sheets, dialogs, toasts with `inert` + focus management). Large results live in
  `app.results`, not in the store. Session metadata (status, stats, thumbnail) is written by the
  app (`SessionStore`) independently of what the engine persists. The three.js viewer is a lazy
  chunk (`viewer-loader.ts`, prefetched when idle): entry chunk ≈ 190 KB.
* **Permissions** (Setup "Start capture" tap): `requestCapturePermissions({geolocation:false})`
  is called synchronously first (motion inside the gesture, then camera), plus wake lock and
  `storage.persist()` in the same tap; geolocation is requested right after the camera opened
  (`GeoSensor.start()`), so an unanswered location prompt / slow first fix never delays the
  camera. Capture: FramePump (10 Hz, back-pressure) → `engine.processFrame`; hidden page ⇒ pause,
  resume only from a tap (camera re-acquired inside it when needed); camera errors / denied
  permissions show recovery cards; the keyframe flash animates only `opacity`; no
  `backdrop-filter` over the live video (re-blurring every video frame competes with engine GPU
  work); mini 3D view ≤ 2 Hz, ≤ 20k points (50k expanded), unanimated fits.
* **Export**: first tap builds the Blob (`engine.exportModel`) and shows its size; the second tap
  calls `saveFile` synchronously in the handler (5 s transient activation for `navigator.share`);
  a `NotAllowedError` in Home Screen mode asks for another tap.
* **Tests**: unit — `format`, `router`, `store`, `overlay-map` (object-fit cover/contain, quarter
  turns), `stages` (name mapping, ETA), `protocol` (transferables, envelopes, errors), `env`,
  `mock-engine` (+ scene geometry, exports decode). Browser — `engine-client.gpu.test.ts` with
  `engine-client.worker-fixture.ts` (transfers, progress routing, cancel, timeouts, respawn +
  resume, CPU fallback, respawn cap, startup failure, visibility, dispose). E2E —
  `node e2e/ui/smoke.mjs` (full mock flow at 393×852 @3x with the fake camera, downloads checked,
  screenshots in `e2e/out/ui/`) and `node e2e/ui/states.mjs` (engine missing, install hint, CPU
  mode, Lockdown, camera denied, landscape, interruption, processing error).
* **Headless caveat**: under SwiftShader, `createImageBitmap(video)` at 10 Hz (and even a bare
  fake-camera `<video>`) saturates the GPU process so that screenshots slow down progressively;
  the e2e scripts call `app.debug.holdFrames(true)` (stops the pump, pauses the `<video>`) around
  capture screenshots. Not an issue on devices.
* **Unverified on device**: wake lock from a tap that navigates to Processing; `navigator.share`
  of 100+ MB exports; Safari's handling of the overlay canvas at DPR 3 during capture; iPad
  landscape layout of the capture controls.

### `src/sfm` core — review notes (adversarial review; amends the `src/sfm` core notes above)

* **Deterministic time-sliced global BA** (`adjustGlobalBundleSliced`): slices used to end on wall-clock after
  any LM iteration and restart with the carried λ but a reset damping-growth factor ν, so batch results depended
  on CPU load (30-image orbit, 3 ms vs 500 ms slices: f 1397.364 vs 1396.695, k1 0.0284 vs 0.0308). Slices now
  end only right after an ACCEPTED step (BA `onIteration` + a private cancel token; accepted ⇔ cost decreased),
  where λ fully describes the LM state, and negative-depth filtering runs only before the first slice ⇒
  bit-identical to one uninterrupted solve (tested with 0.01 ms slices). A slice is still cut after 3 ×
  `globalBASliceMs` during a run of rejected steps (only then can slicing change the result).
  `globalBATimeLimitMs` now bounds the whole sliced BA (was applied per slice, i.e. ignored); the run's cancel
  token is forwarded into BA. Live mode stays time-bounded (load-dependent) by design.
* **Degenerate PnP rejected**: with coinciding / tightly clustered inlier observations PnP + refinement moves the
  camera ~1e8 units away and calls every correspondence an inlier. `registerNextImage` now rejects poses with
  non-finite values or an inlier 2-D RMS spread < `absPoseMaxErrorPx`.
* **Forward motion** (walking straight ahead: every pair |t_z| > 0.95, median angles < 2°) gave NO model in batch
  and live. New last resort `initFallback` (default on), `initFallbackMinTriAngleDeg` 1.5, `initFallbackMaxForwardMotion`
  0.9999: batch adds one level used only when the regular levels found no model (other captures unchanged, tested);
  live relaxes after `liveInitFallbackAfter` (10) keyframes to max(1.5°, ½·`liveInitMinTriAngleDeg`) + forward ≤ 0.9999,
  after 20 to 1.5°; evaluated pairs are cached (`initCache`) and re-checked for free. Forward walk (50–60 frames):
  batch all registered, ATE 6.5e-5 of the extent, f 0.4 %; live initialises at keyframe ≈ 19, all registered.
* **Validation**: `isValidImageId` (integer in [0, 2²⁰)) enforced by `setTwoView`, `CorrespondenceGraph.addImage`
  (also integer feature count ≥ 0) and `Reconstruction.addImage` — `pairKey(0, 2²⁰+5) === pairKey(1, 5)` aliased pairs.
  `addPoint` rejects a track listing an element twice (double-counted `numPoints3D`) or an out-of-range feature, with no
  side effects. `transform` throws for s ≤ 0 / non-finite. `angularError` is π for depth ≤ 0 (was 0 — a "perfect"
  continuation — for a point at the camera centre).
* **Smaller fixes**: `LiveMapper.refine()` reports images dropped by its image filter in `pendingIds()`;
  `findInitialPair` reads stored pairs in canonical order (no inverted copies cached for half the pairs).
* **Clarification**: `merge` (as COLMAP) can join two tracks that both observe an image; continue / complete never add a
  second observation of an image. Visibility counts of images whose graph edges change AFTER they were added (pair
  replacement / late pairs) can go stale; only next-best-view ranking reads them and registration resyncs first.
* Verified correct (no change): `transform` pose update, `invertTwoView`, visibility pyramid vs COLMAP, ray triangulator
  (cheirality, angle tests, LO), local/global gauges (largest relative-translation axis), intrinsics flags, COLMAP
  filter / merge / complete / retriangulate semantics, multi-model registration bookkeeping, cancel at any point.
* **Tests**: `core-review.test.ts` (adversarial: ids, π rotations, 1e±3 scales with 1e5 offsets, parallel / behind rays,
  60 % outlier rays, 600-step bookkeeping fuzz with an exhaustive invariant checker incl. brute-force visibility,
  next-best view, registration thresholds, bit-identical gauges / constant intrinsics, cancel at the k-th token read,
  forward walk, live invariants per keyframe, π-rotation × 1e3 × 1e5 similarity mid-capture), `core-stress.test.ts`,
  `core-testutil.ts` (`reconstructionInvariantErrors`). `synthetic.ts` kinds `'loop'` (closed ≈ 110 m walk around the yard,
  22 × 12 m ellipse, extra south fence / house / tree) and `'forward'`; existing kinds keep their random streams.
* **Measured** (Node, load 8–10): loop 200 keyframes batch 200/200, ATE 3.2e-5 of 93 m (3 mm), max centre error 1.6 cm
  (no loop drift), f 9e-5, k1 0.0299, 0.85 px, ≈ 25 s (+ 13 s verification), max event-loop stall ≈ 0.7–1.3 s; live 200/200,
  per keyframe median 35 ms / p90 100 ms / max 1.3 s, after `refine()` ATE 2.8e-5. Orbit 120 views (31-view tracks, 161 k
  observations): batch 21 s, ATE 3.3e-5, f 1e-5; live 120/120, median 96 ms.

### `src/app` — review notes (adversarial review; amends the `src/app` notes above)

* **EngineClient timeouts pause while the page is hidden.** iOS suspends a hidden page and its
  workers (and drops GPU work), so a request in flight across a trip to the background used to
  "time out" the moment the user came back: 2 frames ⇒ stuck-worker respawn, `init` ⇒ respawn,
  a long refine ⇒ "Processing failed". `setHidden(true)` now stops every request timer and
  `setHidden(false)` restarts the full timeout (progress events received while hidden do not
  re-arm it either).
* **`collectTransferables` no longer transfers partial views** (`pool.subarray(0, n)`, several
  arrays in one allocation) unless `transferPartialViews: true`: transferring detaches the WHOLE
  buffer, so an engine returning a view of its growing point buffer would have silently lost it.
  Partial views are cloned (which copies the full underlying buffer — `slice()` them first).
* **Capture**: frame errors go through `frameErrorAction` (`capture-policy.ts`): a `dead` engine
  (respawn cap) stops the pump with an error card instead of failing every frame at 10 Hz;
  quota errors (`StorageQuotaError` / `QuotaExceededError` / `code: 'quota'`) pause the scan
  with "Finish scan / Resume / Back to home". A quota error when the session record is created
  blocks the scan before any frame ("Storage is full"); < 300 MB of origin quota left ⇒ warning.
  `App.createSession` returns `{status: 'ok' | 'quota' | 'error' | 'unavailable'}`.
* **Overlay**: keypoints of a frame captured at another UI rotation are dropped, not rotated —
  both the frame bitmap and the `<video>` are upright for THEIR rotation (the UI rotation
  compensates the device roll), so a quarter turn put them in the wrong place
  (`overlayMatchesView`). The overlay canvas is capped at 2× DPR (12 MB → 5 MB at DPR 3).
* **Camera reconnect** waits for a restart CameraCapture is already doing after the page became
  visible (`restart()` refused it and showed a bogus "could not restart" card).
* **Processing opened from Home** used the PREVIOUS scan's `app.results` (name, quality, stages
  shown as built) until the engine had loaded the new one; it now shows "Opening scan…" and fills
  in name / quality / stages after `ensureSessionLoaded`. A `dead` engine is not retryable.
* **Export sheet**: a file finished after the sheet was closed and reopened updated the detached
  old list (stuck "Preparing…"); the open sheet's renderer is tracked. Double taps on "Share /
  Save" are ignored for 2 s or until the share settles (a second `share()` rejects with
  `InvalidStateError`, which in a Safari tab fell back to a download on top of the sheet).
* **Navigation**: the exiting screen is `inert` + `aria-hidden` during its fade; focus moves to
  the new screen's `h1` (or the screen) when it was on the old screen / `<body>`; the enter
  transition has a timer fallback when rAF does not run (starved compositor). `beforeunload`
  asks before leaving a running capture / processing (desktop, Android; iOS shows no prompt).
* **Small fixes**: SW registration also when `load` already fired (the `?mock=1` boot awaits a
  chunk); Home thumbnails' object URLs are reused across the per-minute refresh (no flicker) and
  never created after `leave()`; export file names keep non-Latin letters (only Latin accents are
  dropped); `--text-3` raised to ≥ 4.5:1 on every surface; sheet close and toast action buttons
  are 44 px targets; recovery cards scroll (they were clipped in landscape); the expanded live
  3D view in landscape no longer covers Cancel; the live 3D view is never a button inside a
  button; the status pill says "Paused" while paused.
* **Tests**: `capture-policy.test.ts`, `format.test.ts` (non-Latin names), `protocol.test.ts`
  (partial views), `engine-client.gpu.test.ts` (timeouts paused while hidden, no respawn of a
  stuck init while hidden). E2E: `node e2e/ui/states.mjs` gained `storage-full-start`,
  `storage-full-capture`, `engine-dead`, `export-reopen`, `open-other-session`, `focus`;
  `node e2e/ui/layouts.mjs [se|landscape|ipad]` audits every screen and sheet at 375×667,
  852×393 and 1180×820 (no horizontal overflow, controls inside the viewport, ≥ 44 px touch
  targets, no clipped text) with screenshots `e2e/out/ui/layout-*.png`.
* **Still unverified on a device**: Safari's `inert`/focus behaviour with VoiceOver; whether a
  `devicemotion`/camera prompt makes the page `hidden` (it should not); share of 100+ MB files.

### `src/engine` + `src/workers/engine.worker.ts` (PhotogrammetryEngine, worker)

* **Files**: `engine.ts` (PhotogrammetryEngine), `live.ts` (LiveSession: frame analysis, keyframes, mapping
  queue, hand-eye check, 'auto' mode; CaptureState, ImuLog), `frame-ingest.ts` (orientation, frame
  preparation, colours, overlay, gravity), `refine.ts`, `dense.ts` (densify, mesh, PatchMatchHolder),
  `export.ts`, `persistence.ts` (EngineStore: Memory / SessionStore-backed; record formats), `image-cache.ts`
  (byte-bounded LRU, JPEG decode), `depth-store.ts`, `presets.ts`, `cpu-pipeline.ts`, `types.ts`, `index.ts`;
  `testutil.ts` is TEST-ONLY (synthetic orbit captures with exact IMU, not exported).
* **API (ENGINE.md §2, superset)**: `new PhotogrammetryEngine(options, config?)`; `init`, `processFrame`,
  `finishCapture(progress?, cancel?)`, `refine`, `densify`, `mesh`, `exportModel`, `loadSession(id, options?)`
  (works without `init`), `setHidden`, `flush()` (wait for background mapping + writes), `on('live' | 'log' |
  'error', cb)`, `recoverGPU()`, `releaseResultMemory()`, `releasePipeline()`, accessors `liveSession`,
  `sparseModel`, `sparseResult`, `denseCloudResult`, `meshResult`, `engineStore`. `EngineConfig` (tuning:
  injected store / pipeline, SIFT overrides, `mapping: 'background' | 'inline'`, keyframe / live / mapper /
  dense overrides, pair budgets, memory budget) is the 2nd constructor argument, and the worker accepts it as
  the `engineConfig` extension field of `init` / `loadSession` options.
* **SparseCamera extension**: refine / loadSession cameras carry `t` (capture ms) and `frameIndex`
  (FrameInput.index of the source frame, else the engine's 0-based frame counter) — type `SparseCameraEx`
  (`SparseCamera & { t?, frameIndex? }`, protocol.ts unchanged; structured clones keep the fields).
* **No static import of the WebGPU feature modules**: `sift-gpu.ts` / `match-gpu.ts` read `GPUBufferUsage` at
  module load, which throws where WebGPU does not exist (older iOS, Lockdown, Node) and would kill the whole
  worker module. `features/extractor` is imported dynamically only when `navigator.gpu` exists; otherwise
  `cpu-pipeline.ts` (extractSiftCPU + matchCPU) is used.
* **Orientation**: canonical θ₀ = first accepted frame's θ; frames at another θ are rotated (k clockwise
  quarter turns) before analysis; frames < 500 ms after a θ change are dropped; a landscape bitmap in a
  portrait UI is a RAW sensor buffer (upright for θ = 90), a portrait bitmap in a landscape UI is dropped.
  **Hand-eye check**: once ≥ 6 keyframes are registered (and again with the refined poses before the world
  alignment), the visual relative rotations of consecutive keyframes are compared with R_cd(θ)·ΔR_gyro·R_cdᵀ for
  θ ∈ {0, 90, 180, 270}; a clearly better hypothesis (median < 4° and < ½ of the runner-up) replaces
  R_cam_device of every keyframe (FrameInfo.R_cam_device / gravityCam) and is logged. LiveUpdate.overlay is in
  the coordinates of the frame AS SUBMITTED (inverse canonical turn).
* **Live**: per frame ≤ 640 px RGBA copy (sharpness, colours), blur / motion pre-gate (no features for
  rejected frames), SIFT at the preset working size (`extractFromBitmap` with the bitmap or the rotated
  OffscreenCanvas on the GPU, a working-size gray image on the CPU), H-guided matching to the last keyframe
  (H = K R_c K⁻¹ from the gyro, radius 8 % of the long side, unguided fallback below 30 inliers), two-view
  verification (≤ 400 RANSAC iterations), rotation-compensated parallax, KeyframeSelector. Keyframes go to a
  serial **mapping queue** ('background': the frame response returns first, the model update arrives as a
  `live` event; 'inline' for deterministic harness / tests): previous 6 keyframes (4 on the CPU tier) +
  vocabulary retrieval (trained with 256 words after 25 keyframes, then incremental) + spatial neighbours of
  the last registered keyframe (look-at distance, ≤ 45°), each matched (unguided) and verified (≤ 2000 / 1000
  RANSAC iterations; the tracking geometry of the reference pair is reused), DB + graph update,
  `LiveMapper.addKeyframe`. Everything that a respawned worker needs is persisted per keyframe (JPEG q 0.92,
  features, colours, pairs, IMU chunk, GPS, state).
* **Refine**: vocabulary retrained (512 words) when it covered < 50 % of the keyframes; pairs sequential 10 +
  retrieval 15 + spatial 10 (75°) minus every pair already tried; live model: pending keyframes retried, full
  global refinement; offline `runIncrementalSfM` (live-refined intrinsics as the initial guess); the better
  model (registered images, then mean reprojection error) is CLONED (the live model keeps its frame) and
  aligned with `computeWorldAlignment` (camera–IMU offset searched around the last estimate, GPS fixes, compass
  'auto' reference). **Deviation**: when the IMU scale is rejected only for low excitation but the fit is
  excellent (R² ≥ 0.9, σ ≤ 3 %, excitation ≥ 0.03 m/s²) it is accepted with σ = max(2σ, 2 %) and a warning
  (e2e `object` orbit with GT poses: 0.3 % scale error at 0.06 m/s², rejected by the 0.08 floor).
* **Dense**: preset table of ENGINE.md (+ `DENSE_PRESETS` of src/dense when exported, read defensively by
  field name; `removeDepthHalos` applied to the final maps when exported) → evenly spread view subset →
  JPEG decode at the dense size + `undistortImage` through an LRU (15 % of the memory budget) → selectSourceViews
  → PatchMatchGPU (self-tested once; CPU at ≤ 320–480 px otherwise; device loss ⇒ recreate once, else
  'gpu-lost') photometric + geometric passes → fusion (min views 3 object / 2 scene, colour thumbnails within
  8 % of the budget) → SOR. Depth maps live in IndexedDB + LRU: the session store when persisting, otherwise a
  scratch session in the separate database `sfm-scratch` (deleted on dispose, stale ones purged after 1 h).
  Mesh: TSDF on the fusion-filtered maps (object mode: maxDepth = 2 × median sparse depth; block budget ≤ 25 %
  of the memory budget) + optional simplification; after a reload the filtered maps come from IndexedDB.
* **Worker**: one engine per `init` (the feature pipeline is reused across inits with the same SIFT config);
  progress throttled to 10 Hz per request; cancel tokens per request; frame / live results transferred;
  results the engine keeps are cloned, persisted dense clouds / meshes are transferred and dropped from the
  engine (reloaded for export); GPU device loss ⇒ `recoverGPU()`, fatal `gpu-lost` when no adapter is left.
* **Measured** (CPU features in Chromium, inline engine, every frame): e2e `yard` (240 frames) → 80 keyframes,
  80/80 registered, ATE 1.25 cm (3.2e-4 of the extent), scale error −0.08 % (IMU + GPS), gravity 0.017°,
  heading −2.6° (GPS 4-DOF), focal 0.05 %, time offset −24.7 ms (true −25); e2e `object` (200 frames) → 90/90,
  ATE 0.27 mm, scale error +0.4 % (IMU, low-excitation acceptance, σ 2 %), gravity 0.015°, heading −0.54°
  (compass), focal −0.003 %, time offset −25.0 ms. Node unit tests (240×320 synthetic
  orbit): scale < 3 %, ATE < 5 mm, gravity < 1°.
* **Limits / TODO**: no relocalisation into a second live model when tracking is lost for good (offline SfM
  handles disconnected captures); the 'auto' mode decision (optical-axis convergence) is heuristic; principal
  point is not refined (1.8 px on the synthetic data); depth maps of non-persistent sessions without
  IndexedDB stay in memory; on-device (A-series) timings not measured yet.

### `src/engine` + harness — e2e integration notes (amends the `src/engine` notes above)

* **Runs** (`node e2e/run-e2e.mjs <object|yard> --engine worker`, every frame, all sensors, fast preset,
  `--dense-size 480` for SwiftShader; the real worker through `EngineClient`): every check of
  `e2e/thresholds.json` passes on both datasets — numbers in `e2e/README.md` ("Engine end-to-end
  results"). `--engine inline` gives the same accuracy (parity). Feature extraction ran on the CPU
  (`engineConfig.featureBackend: 'cpu'`, below) for the full runs; the GPU SIFT / matcher path through the
  worker (object, every 2nd frame) gives the same accuracy class (ATE 2.0e-4, rotation 0.039°, scale
  +0.86 %), at 25 s per frame on SwiftShader.
* **`EngineConfig.featureBackend: 'auto' | 'cpu'`** (new): CPU SIFT / matching although WebGPU exists,
  PatchMatch stays on the GPU. GPU and CPU SIFT produce the same features (features GPU notes); on
  SwiftShader GPU SIFT costs ≈ 4–8 s per 720×960 frame and GPU matching ≈ 2 s per pair (CPU: ≈ 0.3–0.5 s /
  ≈ 0.05 s), so full captures are only practical with the CPU backend there. Part of `pipelineKey`.
* **Back-pressure of background mapping** (`EngineConfig.maxMappingBacklog`, default 2): a keyframe frame's
  response waits until ≤ 2 keyframes are queued / being mapped. Before, the mapping queue grew without
  bound whenever mapping was slower than the keyframe rate (one JS thread shared with frame analysis):
  at the end of the e2e captures only 31/90 (object) and 36/80 (yard) keyframes were mapped live — the
  user saw a stale model / guidance — and `finishCapture` needed 20–28 s to drain. Now 90/90 and 80/80
  live, `finishCapture` < 1 s; the frame rate drops instead (the FramePump just delivers fewer frames).
* **Principal point refined in the final BA of refine** (`EngineConfig.refinePrincipalPoint`, default: when
  ≥ 20 images are registered): one global BA with (f, cx, cy, k1, k2) free on a clone of the chosen model;
  kept only if the principal point stays within 10 % of the image size of the centre and the mean
  reprojection error does not grow. A fixed, offset principal point is absorbed by a constant rotation of
  every camera: on the e2e datasets (true offset 1.8 px) the mean rotation error drops 0.135° → 0.054°
  (object) and 0.149° → 0.059° (yard), the principal point error 1.80 → 0.3–0.5 px, yard ATE 1.26 → 0.88 cm.
  `stats.principalPoint` says what happened. Mapping (live / offline) still keeps it fixed (COLMAP).
* **`DENSE_PRESETS` option blocks are passed through**: besides the flat numbers, the `patchMatch` (every
  PatchMatch pass), `viewSelection` (over the capture mode's depth margins) and `fusion` blocks of
  `src/dense/presets.ts` now reach the dense module (they were ignored: e.g. the tuned
  `viewSelection.depthMarginMin` 0.5 and `fusion.maxNormalErrorDeg` 15), and `haloFilter: false` in the
  preset disables the engine's halo pass. Fusion never gets its own halo filter when the engine already
  applied `removeDepthHalos` to the depth maps (the preset's "not both" rule).
* **Harness fixes** (`src/harness/engine`): the compass feed emulates `CompassSensor` (device-top azimuth
  while the top is within 45° of horizontal, else the optical axis — the `compassReference: 'auto'` the
  world alignment assumes; the dataset's optical-axis `heading` read as a device-top heading is 90° off in
  the landscape yard — with `--no-gps` the yard heading error is now −0.15°); the live summary is refreshed
  after `finishCapture` (it was taken before the mapping queue drained). `loadSynthDataset` treats an
  optional file answered with Vite's index.html as missing (object has no geo.json). `e2e/ui/states.mjs`
  `engine-missing` serves a stub worker now that the real worker exists.
* **Remaining** (by impact): yard dense completeness 0.38 at τ = 0.1 m (grazing ground, far houses at
  480 px; GT-pose reference 0.23–0.34); yard heading −2.7° = GPS 4-DOF yaw (the compass is not fused when
  GPS decides the yaw — and must not be until the app applies the magnetic declination:
  `CompassSensor.setDeclination` is never called, so compass headings on a device are MAGNETIC, e.g. ≈ +15°
  off in Seattle); dense timings on SwiftShader (≈ 25–45 s per 480 px view) say nothing about Apple GPUs.

### `src/dense` — Dense tuning results (GT-pose harness, 2026-10)

* **Setup**: `e2e/run-dense-gt.mjs` with exact poses, so every error is MVS or fusion. Reference
  ("full") runs: 480 px; object 24 views (360×480), yard 30 views (480×360), stride 8; 8 sources;
  6 photometric + 3 geometric iterations. τ = `denseTau` (object 13.6 mm, yard 0.1 m); P / C / F =
  precision / completeness / F-score at τ, identity alignment. Fusion experiments re-fuse the
  saved reference maps (`--load-maps`, seconds each). PatchMatch experiments run on "mini"
  subsets at 480 px (yard frames 0–112 = 15 views, object frames 0–88 = 12 views). Each is
  re-fused with the reference and with the tuned fusion, and compared on Csub, the completeness
  of the GT samples visible in the used views. SwiftShader was shared with other jobs (load 5–12),
  so timings are relative only. One change at a time; a change was kept only when it helped object
  AND yard without costing the other metric materially.
* **API additions (backward compatible)**:
  * `depth-filter.ts`: `depthHaloMask(dm, image, cam, opts?)` returns a 1-bit mask;
    `removeDepthHalos(dm, image, cam, opts?)` returns a cleaned copy; `DEFAULT_HALO_FILTER_OPTIONS`.
  * `FusionOptions.haloFilter`: the same filter inside fusion (default off). `FusionStats` gains
    `haloRejected` / `haloMs`.
  * `presets.ts`: `DENSE_PRESETS`, `DensePreset`, `DenseQuality`.
  * Harness: `--preset`, `--geom-passes`, `preHalo`, `filter`, and halo-rejection statistics per
    GT error class.
* **Fusion experiments** (full maps; P / C / F):

  | # | Setting | object | yard |
  |---|---|---|---|
  | 0 | reference: `minViews` 3, normal 10° | .9991 / .9601 / .9792 | .9885 / .2307 / .3741 |
  | 1 | `minViews` 2 | .9982 / .9875 / .9928 | .9777 / .3376 / .5019 |
  | 2 | 1 + normal 20° · 30° · off | .9978 / .9889 / .9934 · .9976 / .9898 / .9937 · .9965 / .9924 / .9944 | .9641 / .3599 / .5242 · .9541 / .3691 / .5323 · .9122 / .3846 / .5411 |
  | 3 | 1 + low-angle rule (largest triangulation angle < 3° · 5° ⇒ 3 views) | .9982 / .9875 / .9928 · .9983 / .9875 / .9929 | .9846 / .3169 / .4795 · .9879 / .2964 / .4560 |
  | 4 | 1 + halo filter, colour-mean rule, r 4 (dropped) | .9986 / .9867 / .9926 | .9802 / .3349 / .4993 |
  | 5 | 1 + halo filter, vote rule, r 4 · r 8 step 2 | .9985 / .9865 / .9925 · .9996 / .9872 / .9933 | .9809 / .3355 / .5000 · .9828 / .3343 / .4989 |
  | 6 | 5 (r 8) + normal 12° · **15°** · 20° | .9997 / .9877 / .9936 · .9995 / .9882 / .9939 · .9995 / .9885 / .9939 | .9814 / .3402 / .5053 · .9780 / .3476 / .5129 · .9743 / .3556 / .5211 |
  | 7 | 1 + normal 15°, no halo filter | .9980 / .9886 / .9933 | .9705 / .3515 / .5160 |
  | 8 | **6 (15°) with halo step 4** (3× cheaper, same result) = tuned fusion | .9996 / .9883 / .9939 | .9788 / .3473 / .5127 |
  | 9 | 8 + `maxDepthError` 0.02 | .9996 / .9887 / .9941 | .9771 / .3520 / .5175 |
  | 10 | 8 + `maxNeighbours` 6 · 8 (reference 10) | .9997 / .9870 / .9933 · .9997 / .9880 / .9938 | .9794 / .3473 / .5127 · .9788 / .3475 / .5129 |
  | 11 | 8 + SOR k 8, σ 1.5 · 2.5 · 4 | .9998 / .9844 / .9920 · .9998 / .9865 / .9931 · .9997 / .9877 / .9937 | .9838 / .3282 / .4922 · .9816 / .3386 / .5035 · .9801 / .3438 / .5090 |
  | 12 | 8, halo filter applied twice (map + fusion) | .9998 / .9880 / .9939 | .9819 / .3440 / .5095 |
  | 13 | 8 on photometric maps (no geometric pass) | .9997 / .9880 / .9938 | .9821 / .3351 / .4997 |
  | 14 | 8 on yard maps computed with `depthMarginMin` 0.5 | — | .9809 / .3522 / .5183 |

  TSDF mesh of the fusion-filtered maps, P / C / F: object .990 / .938 / .963 → .999 / .974 / .987
  (tuned); yard .898 / .216 / .348 → .844 / .340 / .484. Building the TSDF input with `minViews` 3
  (cloud with 2) is worse on both: object .9998 / .943 / .970, yard .883 / .245 / .384.
* **PatchMatch experiments** (mini subsets; Csub / P with the reference fusion → with the tuned
  fusion; "≤1 % / >5 %" = geometric-pass pixels within 1 % / off by more than 5 % of the surfel
  GT; PM s = PatchMatch seconds, contended):

  | run | yard Csub / P: ref fusion → tuned | yard ≤1 % / >5 % | object Csub / P: ref fusion → tuned | object ≤1 % / >5 % | PM s yard / object |
  |---|---|---|---|---|---|
  | reference settings (8 sources, 6 + 3 iterations, margin 0.8) | 0.3083 / 0.9905 → 0.5234 / 0.9797 | 84.2 / 9.8 | 0.8296 / 0.9999 → 0.9166 / 0.9997 | 84.4 / 8.7 | 726 / 969 |
  | `depthMarginMin` 0.5 (base of the rows below) | 0.3076 / 0.9906 → 0.5277 / 0.9810 | 87.7 / 6.5 | 0.8235 / 1.0000 → 0.9167 / 0.9998 | 84.0 / 9.3 | 914 / 1214 |
  | multi-scale (½-res 6 it → full-res 2 it prior → geometric) | 0.3102 / 0.9899 → 0.5286 / 0.9795 | 88.2 / 6.2 | 0.8209 / 0.9999 → 0.9151 / 0.9996 | 84.8 / 8.4 | 823 / 643 |
  | 5 sources | 0.3041 / 0.9903 → 0.5253 / 0.9810 | 87.6 / 6.5 | 0.8176 / 1.0000 → 0.9085 / 0.9995 | 83.0 / 9.9 | 686 / 702 |
  | 4 photometric iterations | 0.3065 / 0.9909 → 0.5242 / 0.9811 | 87.8 / 6.4 | 0.8223 / 1.0000 → 0.9151 / 0.9998 | 83.9 / 9.5 | 767 / 754 |
  | 2 geometric iterations | 0.3087 / 0.9906 → 0.5287 / 0.9809 | 87.8 / 6.4 | 0.8223 / 1.0000 → 0.9155 / 0.9998 | 84.1 / 9.3 | 970 / 1178 |
  | 2 geometric passes | 0.3139 / 0.9897 → 0.5299 / 0.9796 | 88.0 / 6.5 | 0.8323 / 1.0000 → 0.9157 / 0.9997 | 84.7 / 9.2 | 1066 / 1160 |

* **Kept**:
  1. `minViews` 2 for object and scene mode: the largest gain. Keyframe-spaced views rarely give
     3 agreeing maps; with GT depth fed into fusion, C goes 0.316 → 0.461 on yard and
     0.983 → 0.998 on object.
  2. The colour-guided halo filter (`depthHaloMask` / `removeDepthHalos`, `depth-filter.ts`) on
     the final maps. It rejects 66–68 % of the > 5 % outliers and 81 % of the sky estimates, for
     0.9–1.2 % of the ≤ 1 % pixels. With `minViews` 2 and 10°, yard points farther than τ fall
     from 21.4 k to 16.3 k, and 0–10 m precision is 99.8–100 %. Object points farther than τ fall
     from 388 to 192 in the final run, which removes most of the visible beige silhouette halos. Cost ≈ 45 ms
     per 480 px map.
  3. Normal threshold 15° with the halo filter. PatchMatch normals of far and grazing surfaces
     are noisy, and 15° buys back the completeness the filter costs (row 8 vs row 1: object
     +0.14 P / +0.08 C, yard +0.11 P / +0.97 C).
  4. `depthMarginMin` 0.5 in view selection. The 1st-percentile sparse depth misses the near
     ground in 25 of 30 yard views: the GT sparse samples are area-uniform, and real SfM tracks
     are short there. With the wider bound, the yard maps lose a third of their > 5 % outliers
     (11.5 → 7.7 %) and 0–5 m completeness rises from 56 to 68 %. On object it is neutral with the
     tuned fusion (Csub 0.9166 → 0.9167), but −0.6 pt with the reference fusion.
  5. Cheaper PatchMatch for `fast`, measured with the tuned fusion (object / yard Csub):
     * 4 photometric iterations: −0.16 / −0.35 pt;
     * 5 sources: −0.8 / −0.24 pt, and −25–38 % PatchMatch time;
     * 2 geometric iterations: −0.12 / +0.10 pt.
* **Not kept**:
  * low-angle rule: trades far-field completeness for precision at about 3:1;
  * SOR: it removes sparse but correct far points (yard −0.9 pt C for +0.3 pt P at σ 2.5), so the
    presets set `sorK` 0;
  * `maxDepthError` 0.02: yard +0.5 C / −0.2 P, object ±0, too small to decide on;
  * applying the halo filter twice: it erodes edges (yard −0.3 pt C);
  * multi-scale: quality-neutral (yard Csub +0.1 / P −0.15, object −0.16 / −0.02), but 25–45 %
    cheaper in the photometric stage. The engine has no multi-scale driver, so `multiScale: false`;
  * a second geometric pass: +0.6 / +0.9 pt Csub with the reference fusion, but within ±0.2 pt
    with the tuned one (yard +0.2 / P −0.14, object −0.1), for +33 % PatchMatch time. `max`
    therefore keeps one pass.

  The function defaults (`DEFAULT_FUSION_OPTIONS`, `DEFAULT_VIEW_SELECTION_OPTIONS`,
  `DEFAULT_PATCHMATCH_OPTIONS`) are unchanged, because existing tests pin them (`minViews` 2 or
  15° breaks `fusion.test` / `dense-e2e.test`). The tuned values live in `DENSE_PRESETS`.
* **Final vs reference** (full views, 480 px, `--preset balanced`):

  | | object P / C / F, acc med / p90, mesh F | yard P / C / F, acc med / p90, mesh F |
  |---|---|---|
  | reference | 0.9991 / 0.960 / 0.979, 0.11 / 0.59 mm, 0.963 | 0.9885 / 0.231 / 0.374, 3.4 / 31 mm, 0.348 |
  | tuned | 0.9997 / 0.987 / 0.994, 0.13 / 0.63 mm, 0.987 | 0.9809 / 0.352 / 0.518, 2.9 / 33 mm, 0.492 |

  Yard by distance to the nearest camera, C (P): 0–5 m 31 → 68 % (100 → 100 %), 5–10 m
  52 → 76 % (99.9 → 99.8 %), 10–20 m 42 → 61 % (99.4 → 97.9 %), 20–40 m 24 → 36 % (94 → 88 %),
  > 40 m 3.6 → 6.2 % (65 → 53 %).

  The precision cost of `minViews` 2 is far-field depth noise at 480 px: the wrong far points are
  1.6–1.8 τ off (median), not gross errors. A sky-coloured fringe also remains on roof and tree
  outlines (28 % of the remaining > τ points); a third of the sky-coloured points are within τ.

  The object's accuracy median and p90 rise slightly (0.11 → 0.13 mm and 0.59 → 0.63 mm) because
  2-view points are noisier. Precision at τ rises, and the focus box stays at 99.96 %.
* **`DENSE_PRESETS`** (`src/dense/presets.ts`, exported from `src/dense`;
  `src/engine/presets.ts` reads the flat fields). Shared by all presets: halo filter on,
  `minViews` 2 for object and scene, normal 15°, `depthMarginMin` 0.5, SOR off, no multi-scale.

  | preset | dense max side | max views | sources | photometric it | geometric passes × it | fusion neighbours | work per view (MP·src·it) | SwiftShader est. per view | iPhone est. (30–100×) |
  |---|---|---|---|---|---|---|---|---|---|
  | `fast` | 800 | 40 | 5 | 4 | 1 × 2 | 6 | 14 | ≈ 45 s | 0.5–1.5 s per view, ≤ 1 min for 40 views |
  | `balanced` | 1280 | 80 | 8 | 6 | 1 × 3 | 8 | 89 | ≈ 4.7 min | 3–9 s per view, 4–13 min for 80 views |
  | `max` | 1600 | 150 | 8 | 6 | 1 × 3 | 8 | 138 | ≈ 7.4 min | 4.5–15 s per view, 11–37 min for 150 views |

  Image sizes and view counts are the engine's earlier values. They are kept because resolution
  and views are the levers for far content (GT-depth bounds above). On SwiftShader they could
  only be checked at 480 px.

  The engine reads the flat fields: `maxImageSize`, `maxViews`, `numSources`, `iterations`,
  `geometricPasses`, `geometricIterations`, `cpuMaxImageSize`, `fusionMinViews*`,
  `fusionMaxNeighbours` and `sorK` / `sorStd`. It also applies `removeDepthHalos`. It does NOT
  yet pass `fusion.maxNormalErrorDeg` (15°) to `fuseDepthMaps`, nor `viewSelection.depthMarginMin`
  (0.5) to `selectSourceViews` (its `modeConfig` uses 0.8 / 0.7). To get the measured result it
  should spread `DENSE_PRESETS[q].fusion` and `.viewSelection` into those calls.

  Cost model: PatchMatch time ∝ MP × sources × (photometric + geometric iterations), where one
  geometric iteration costs about the same as a photometric one. On SwiftShader (4 cores,
  uncontended) this is ≈ 3.2 s per MP·source·iteration. An iPhone GPU is assumed to be
  30–100× faster; that is not measured. Fusion plus the halo filter take ≈ 0.6 µs per pixel per
  view on one x86 core. Memory: fusion keeps (`fusionMaxNeighbours` + 2) × 20 B/px resident,
  which is ≈ 370 MB at 1600 px with 8 neighbours. That is why `max` uses 8 neighbours, not 10:
  row 10 shows no quality loss.
* **Far vs near range**: nothing in the pipeline assumes a depth scale:
  * depth ranges are per view, with inverse-depth sampling;
  * the fusion tests are relative (1 % depth, 2 px, 15°);
  * the halo filter works in pixels and relative depth;
  * the TSDF voxel is automatic.

  The yard houses at 20–50 m are resolution-limited at 480 px: the pixel footprint is ≈ τ at
  40 m, and GT depth fed into fusion reaches only 14 % completeness beyond 40 m at 480 px and 21 %
  at 960 px. That is why the presets keep 800–1600 px. The near ground (2–5 m) needed the wider
  near bound.
* **Remaining issues**:
  * Far-field precision with 2-view points depends on resolution and view count. More keyframes
    in large scenes help most: with GT depth, 60 instead of 30 yard views raises C from 0.461 to
    0.570.
  * Sky halos wider than the 8 px window survive, because plane propagation extends roofs and
    trees into the sky consistently across views. A PatchMatch-side fix is still needed, such as a
    centre-segment texture test.
  * Low-texture walls in the object room stay sparse.
  * The engine has no multi-scale driver.
  * The yard TSDF mesh loses precision to the noisier far points.
  * Not tested: topK, the ACMH thresholds, the perturbation schedule, sigmaColor, window size,
    and relaxing source selection at trajectory turns (yard frame 184 keeps 3 sources).
  * Timings on Apple GPUs are still unmeasured.
