# Synthetic datasets and evaluation

Rendered image sequences with exact ground truth (poses, intrinsics, IMU, GPS, compass,
depth, dense surface samples) for headless end-to-end tests of the whole pipeline
(features → SfM → inertial/GPS alignment → dense), and a scorer for the results.

* Browser code: `src/harness/synth/**` (three.js, WebGL2), page `synth.html`.
* Generator: `node e2e/generate-synth.mjs` (Vite dev server + headless Chromium/SwiftShader).
* Scorer: `node e2e/metrics.mjs` (plain Node, no browser).
* Datasets land in `e2e/data/<name>/`, which is gitignored. The Vite dev server also serves
  them at `/e2e/data/<name>/…`, and `src/harness/synth/dataset.ts` provides a typed
  `loadSynthDataset(url)` for browser harnesses.

## Generating

```sh
node e2e/generate-synth.mjs [object|yard|all] [--frames N] [--fps F] [--size WxH]
     [--duration S] [--out DIR] [--name NAME] [--seed N] [--blur N] [--no-noise]
     [--depth-every N] [--gt-points N] [--gt-focus-points N] [--time-offset MS]
     [--texture-scale X] [--visibility-every N] [--supersample N] [--set key=json ...]

node e2e/generate-synth.mjs all --frames 40     # quick: 40 frames spread over each trajectory
node e2e/generate-synth.mjs yard                # full default: 240 frames (6 fps × 40 s)
```

* Defaults: `object` 720×960 portrait (θ = 0), 8 fps × 25 s; `yard` 960×720 landscape
  (θ = 90), 6 fps × 40 s.
* Frame counts: `--frames N` alone spreads N frames over the scene's default duration
  (fps = N / duration). `--frames` with `--fps` sets duration = N / fps. `--fps` alone gives
  N = fps · duration.
* `--set` overrides any `SynthConfig` field by dotted path with a JSON value, for example
  `--set effects.vignetting=0 --set imu.gyroBias=[0,0,0] --set gps.correlationS=20`.
  `--set gps=null` disables GPS for the yard.
* `--blur N` turns on motion blur: N sub-frame renders across the exposure (12 ms for
  object, 4 ms for yard). `--no-noise` disables sensor noise.
* Timing (4 CPU cores, SwiftShader): about 0.75 s per colour frame for object and
  0.9 s for yard, plus about 7 s to build the scene and about 0.1 s per frame for
  visibility. Measured runs:

  | Run | Frames | Time | Size |
  |---|---|---|---|
  | object, 40 frames | 40 | 46 s | 44 MB |
  | yard, 40 frames | 40 | 59 s | 44 MB |
  | object, full default | 200 | 188 s | 93 MB |
  | yard, full default | 240 | 248 s | 107 MB |

  Most of the size of the small runs is GT samples and depth maps. A JPEG is about
  0.15–0.25 MB.

## Dataset layout (`e2e/data/<name>/`)

| File | Content |
|---|---|
| `meta.json` | `name, scene, width, height, fps, numFrames, duration, startTimeMs, intrinsics {f,cx,cy,k1,k2,width,height}, screenAngle, R_cam_device (row-major 9), worldFrame, gravityWorld, originLatLonAlt (yard) / null, sceneExtent, sceneBounds, roi, focus, denseTau, timeOffsetMs, imuNoise, gpsNoise, headingNoise, render, depth, gt {...}, seed, config, timingsMs` |
| `frames/NNNNNN.jpg` | JPEG q = 0.92, upright for the UI orientation θ (`R_cam_device` applies) |
| `frames.json` | `[{index, file, t, tTrue, pose: {R: number[9], t: number[3]}, center, gain, depthFile?}]` |
| `imu.json` | `ImuSample[]` at 60 Hz like iOS DeviceMotion (`t, gyro, accel, accelG`), from 0.5 s before the first frame to 0.5 s after the last (`--set imu.rateHz=100`, `--set imu.timeJitterMs=2` for delivery jitter) |
| `heading.json` | `[{t, heading, accuracy, headingDeviceTop, trueHeading}]` at 10 Hz |
| `geo.json` | yard only: `GeoFix[]` at 1 Hz (+ `trueEnu`) |
| `depth/NNNNNN.f32` | every `depthEvery`-th frame (default 20): Float32 LE, W×H, row 0 = top. Each value is the z-depth in metres of the surface at that pixel centre of the distorted JPEG; 0 means no surface. |
| `gt_points.bin` | area-uniform surface samples of the whole scene, Float32 xyz (default 1 M object, 2 M yard) |
| `gt_normals.i8` | face normals of those samples, int8 xyz = round(127·n) |
| `gt_visibility.u8` | number of frames that see each sample unoccluded (saturating at 255) |
| `gt_focus_*.{bin,i8,u8}` | object only: 1 M extra area-uniform samples over the object of interest (object, pedestal, table top, placemat, magazine, mug, pen) |

### Conventions (docs/ARCHITECTURE.md §2)

* World: Z-up ENU (x east, y north, z up), metres. For the yard the ENU origin is
  `originLatLonAlt` (WGS84 47.6062, −122.3321, 56 m).
* Poses are world-to-camera, `x_cam = R X + t`, in the OpenCV camera frame. Matrices are row-major.
* Camera model RADIAL, with pixel centres at integer coordinates. The images follow it
  exactly: the frame is rendered undistorted over a larger window at 2× resolution and
  resampled through the inverse distortion. Defaults are hfov 65° on the long side
  (f ≈ 753.4 px), k1 = −0.03, k2 = 0.005, and a principal point 1–2 px off centre.
* `R_cam_device(θ) = diag(1,−1,−1)·Rz(θ)`, `R = R_cam_device·R_wdᵀ`, and the camera and IMU are co-located.
* IMU signs: gyro = ω_dev with dR_wd/dt = R_wd [ω]×. accel = R_wdᵀ a_world, and
  accelG = R_wdᵀ (a_world − g) with g = (0, 0, −9.81), so a phone flat and face-up reads
  accelG = (0, 0, +9.81).
  Noise: white, σ = 0.004 rad/s and 0.04 m/s² per sample, plus constant biases
  (`meta.imuNoise`). The same accelerometer noise is applied to `accel` and `accelG`.
* Time: milliseconds. IMU, GPS and heading use the true clock. Frames report
  `t = tTrue + timeOffsetMs` (default +25 ms), so a pipeline that estimates the camera–IMU
  offset should recover +25 ms.
* GPS: σ_h(2-D RMS) = 3 m, σ_v = 5 m, white noise by default (`gps.correlationS` for Gauss–Markov).
* Heading: degrees clockwise from true north of the camera optical axis projected to
  horizontal, white noise with σ = 5°.

### Scenes

* object: a torus knot on a marble pedestal, a globe and a printed box (about 30 cm in
  all), on a placemat on a 1.2 × 0.8 m wooden table (top at z = 0.75). The room is
  5 × 5 × 2.6 m with a plank floor, a rug, plaster walls with posters, a window, a door,
  a bookshelf, a couch and a plant. Trajectory: a hand-held orbit with radius
  0.45 ± 0.04 m and 1.2 revolutions in 25 s. The camera sits 10–30 cm above the object
  centre and looks at a wandering aim point, with tremor and roll.
* yard: 120 × 120 m textured ground (macro map plus two tiled detail maps), a fenced
  28 × 27 m yard, a gravel path, a patio with a picnic table, a garden bed, a sandbox and
  a shed. There are 8 houses with brick, siding, stucco or stone walls, windows, doors
  and gable roofs at 18–47 m, plus about 60 trees and bushes. Trajectory: a walk of
  1.03 laps of a 15 × 10 m loop at about 1 m/s, phone at 1.5 m with a 2 Hz ±3 cm bob and
  ±2 cm sway, looking outward with ±25° slow panning.

All materials are unlit, and the shading is Lambertian, baked and view-independent, so
multi-view photo-consistency holds exactly. On top come realism effects: seeded sensor
noise (shot + read, in linear light), 18 % corner vignetting, ±5 % auto-exposure drift,
optional motion blur, and JPEG compression.

## Scoring

```sh
node e2e/metrics.mjs <datasetDir> <result.json> [--tau T] [--out metrics.json] [--no-roi]
     [--min-visibility N] [--dense-align sim3|sim3+icp|se3|none] [--max-points N]
```

Result JSON:

```json
{ "intrinsics": {"f":753.4,"cx":360.9,"cy":478.5,"k1":-0.03,"k2":0.005},
  "images": [{"index": 0, "pose": {"R": [9 numbers, row-major], "t": [3 numbers]}}],
  "metric": {"metric": true, "gravityAligned": true, "northAligned": false},
  "pointsFile": "points.ply", "numInputImages": 120 }
```

`index` is the frame index in `frames.json`. Poses are world-to-camera in the result's own
world frame. `pointsFile` is a binary or ASCII PLY (vertex x/y/z of any type), or raw
Float32 xyz, resolved relative to the result file. `numInputImages` is optional: the
number of keyframes the pipeline attempted.

Metrics (printed as a table and written to `metrics.json`):

* registration: registered / frames (and / `numInputImages`).
* trajectory: Umeyama Sim3 on the camera centres. Reports ATE RMSE, median and max (m),
  ATE / `sceneExtent`, rotation error after alignment (mean, median, max), and the mean
  and max relative rotation error between consecutive registered images.
* metric (only when `metric.metric`): Sim3 scale `s` (maps result → GT), `scaleError = 1/s − 1`
  (relative scale error of the result, + = too large), and ATE after
  an SE3 alignment (no scale).
* frame: the rotation that maps the result world frame to GT is the chordal mean of
  `R_gtᵀ R_res` over images. Gravity error is the angle between the result +Z and GT +Z.
  Heading error is the signed angle of the result +Y (north) about GT +Z, positive when it
  is rotated clockwise. Both are always computed, and the flags say which ones were claimed.
* intrinsics: focal error %, principal point error (px), k1 and k2 errors, and the
  difference in radial displacement at the image corner (px; both models evaluated on the
  GT ray through the corner pixel).
* dense (when `pointsFile` is set): points are aligned with the camera Sim3 (add
  `--dense-align sim3+icp` for a trimmed point-to-point ICP refinement; `se3` uses a
  separate rigid Umeyama fit of the camera centres, so a metric result keeps its own scale)
  and restricted to `meta.roi`.
  * Accuracy is the distance to the nearest GT sample (base + focus sets, k-d trees),
    refined point-to-plane with the sample normal. Reported as median and p90.
  * Precision is the fraction of points within τ.
  * Completeness is the fraction of visible GT samples with a result point within τ.
    It uses the focus set when present, otherwise the base set inside the ROI.
  * F-score combines precision and completeness.
  * τ defaults to `meta.denseTau`: 1 % of `sceneExtent` for object (about 1.35 cm), 0.1 m for yard.

Library use: `import { evaluate, computeMetrics, formatTable, loadDataset } from './e2e/metrics.mjs'`.
Typed helpers for tests live in `e2e/lib/*.mjs` + `*.d.mts`.

## Dense MVS with ground-truth poses

`node e2e/run-dense-gt.mjs` checks the dense stage (src/dense) on its own. It uses the exact
poses and intrinsics, so every error it reports comes from MVS or fusion, not from SfM.

```sh
node e2e/run-dense-gt.mjs [object|yard|all ...] [--max-size 480] [--views N] [--stride N] [--frames i,j,...]
     [--preset fast|balanced|max] [--sources 8] [--iterations 6] [--geom-iterations 3] [--geom-passes 1]
     [--no-geometric] [--multiscale] [--tsdf] [--save-maps] [--load-maps RUN] [--fusion-input geo|photo]
     [--oracle] [--tag NAME] [--set key=json ...] [--no-render] [--no-metrics] [--no-consistency]

node e2e/run-dense-gt.mjs all --tsdf --save-maps            # reference run (~1.5 h on SwiftShader)
node e2e/run-dense-gt.mjs object --load-maps object --tag mv2 --set fusion.minViews=2   # fusion only, seconds
node e2e/run-dense-gt.mjs yard --oracle --tag oracle      # fuse GT depth maps: upper bound without MVS errors
node e2e/run-dense-gt.mjs all --preset balanced --tag bal  # tuned DENSE_PRESETS settings (image size stays 480 px)
node e2e/run-dense-gt.mjs all --load-maps {scene} --tag t --set fusion.minViews=2 --set preHalo=true \
     --set 'filter={"statistical":{"k":8,"std":2.5}}'     # halo filter + SOR on saved maps
```

`--preset` applies `DENSE_PRESETS[preset]` of src/dense (sources, iterations, geometric passes,
view selection, fusion incl. the object / scene `minViews`, halo filter, SOR); explicit options
and `--set` override it, while the image size and view count stay the harness's (`--max-size`,
`--views`). Config keys reachable with `--set` (see `DenseGTConfig` in
`src/harness/dense-gt/main.ts`): `pm.*`, `geo.*`, `selection.*`, `fusion.*` (incl.
`fusion.haloFilter={}`), `preHalo` (`removeDepthHalos` on the fusion input, as the engine does),
`filter` (`filterPointCloud` on the fused cloud), `geometricPasses`, `multiScale`. When a halo
filter is configured the run also prints which GT error classes it rejects.

The page is `dense-gt.html`, and the logic is in `src/harness/dense-gt/`. `window.runDenseGT(config)` does the following:

1. It takes `maxViews` frames at a fixed stride. The defaults are 24 for object and 30 for yard,
   spread over the whole trajectory.
2. It decodes the JPEGs with `createImageBitmap` and an `OffscreenCanvas`, then undistorts them
   with `undistortImage` to `maxSize`.
3. It builds the views with the GT poses.
4. It renders a **GT depth map for every view** with a surfel z-buffer. Each GT sample becomes a
   disc of radius 1.3 × spacing in the plane of its normal. When a focus set exists, base samples
   next to it are dropped. Against the dataset's exact depth maps, these surfel maps agree to
   within 1 % on 98.9 % of pixels.
5. It derives **sparse points** for `selectSourceViews`. It draws 30 000 random GT samples and
   keeps those visible in at least 2 views. A sample is visible when it is no more than
   max(1 %, 1.5 r) behind the surfel z-buffer.
6. It runs `PatchMatchGPU`: a photometric pass, or the multi-scale recipe with `--multiscale`,
   followed by a geometric pass.
7. It runs `fuseDepthMaps`, then `tsdfMesh` on the fusion-filtered maps when `--tsdf` is set.

Output goes to `e2e/out/dense-gt/`:

* `<name>.ply`: the fused cloud.
* `<name>.mesh.ply`: the mesh.
* `<name>.result.json`: input for `metrics.mjs`.
* `<name>.stats.json`: per-view PatchMatch timings and per-stage timings. It also holds the
  per-view depth accuracy of the photometric, geometric and fused maps. Each is measured against
  the surfel GT and, where the dataset has one, the exact GT. The file also holds the dense
  metrics.
* `<name>/debug/view-*.png`: panels showing the image, GT depth, depth, error, NCC cost and
  fused pixels.
* `<name>/render-*.png`: viewer screenshots.
* `<name>/maps/*.f32`: the depth maps, written with `--save-maps`.

Scoring uses `metrics-core` with `denseAlign: 'none'`, plus these breakdowns:

* completeness restricted to the GT samples visible in the used views;
* accuracy and completeness by distance to the nearest camera;
* for object, precision inside and outside the focus box.

Reference run (2026-10): 480 px; 24 views for object and 30 for yard; 8 sources; 6 photometric
and 3 geometric iterations; default fusion; τ = `denseTau`:

| scene | precision | completeness | F | same with `fusion.minViews=2` | oracle (GT depth), minViews 3 / 2 |
|---|---|---|---|---|---|
| object | 0.999 | 0.960 | 0.979 | P 0.998, C 0.988, F 0.993 | C 0.983 / 0.998 |
| yard | 0.989 | 0.231 | 0.374 | P 0.978, C 0.338, F 0.502 | C 0.316 / 0.461 |

Tuned (`--preset balanced`, same views and 480 px: view-selection near margin 0.5, fusion
`minViews` 2, normal 15°, `removeDepthHalos` on the final maps; see ARCHITECTURE §6 "Dense tuning
results" for every experiment): object P 0.9997, C 0.987, F 0.994 (mesh F 0.963 → 0.987); yard
P 0.981, C 0.352, F 0.518 (mesh F 0.348 → 0.492).

## Engine end-to-end runs (`run-e2e.mjs`)

`node e2e/run-e2e.mjs` replays a dataset through the whole engine like the app does, then scores the
result. It starts Vite and headless Chromium (SwiftShader WebGPU), opens `harness.html` once per
dataset and calls `window.runPipeline(config)`. The browser code is in `src/harness/engine/`.

```sh
node e2e/run-e2e.mjs [object|yard|all|<dataset> ...] [--engine worker|inline|mock]
     [--quality fast|balanced|max] [--mode object|scene|auto] [--stride N] [--start N]
     [--max-frames N] [--fps F] [--no-imu] [--no-gps] [--no-heading] [--hfov DEG|gt|none]
     [--no-dense] [--no-mesh] [--dense-size N] [--dense-views N] [--mesh-voxel V]
     [--mesh-triangles N] [--no-export | --export fmt,fmt] [--mapping inline|background]
     [--timeout-min M] [--out DIR] [--tag NAME] [--thresholds FILE] [--no-thresholds]
     [--enforce] [--mock-speed X] [--set key=json ...] [--quiet] [--verbose] [--headed]

node e2e/run-e2e.mjs all --engine mock                                  # plumbing check, ~40 s
node e2e/run-e2e.mjs object --stride 4 --max-frames 40 --no-dense       # short real run
node e2e/run-e2e.mjs yard --dense-size 480 --dense-views 30 --timeout-min 240
```

Defaults: dataset `object`, engine `worker`, quality `fast`, every frame, all sensors, dense, mesh
and every export. `--engine worker` uses `EngineClient` over `src/workers/engine.worker.ts`, which
is what the app uses. `inline` runs `PhotogrammetryEngine` from `src/engine` on the page. `mock` runs
the app's `MockEngine`, whose results are meaningless.

### What the harness feeds

The frames go in time order. `--start`, `--stride` and `--max-frames` select them. `--fps` adds a
FramePump-style rate limit: a frame is dropped when it comes less than 1/F after the last one,
with half a dataset frame interval of tolerance. Each frame is a `FrameInput` with these fields:

* `bitmap`: `createImageBitmap(jpegBlob)`. The next frame is decoded while the engine works.
* `t`: the dataset `t`, which includes the +25 ms camera–IMU offset. `screenAngle` is `meta.screenAngle`.
* `imu`: the samples with `t` in (previous delivered frame's `t`, `t`]. The first frame gets the
  last second.
* `geo`: the latest GPS fix at or before `t`, without `trueEnu`. `heading`: the latest compass
  sample at or before `t`, as `CompassSensor` would report it: `headingDeviceTop` while the device
  top is within 45° of horizontal (gravity from the IMU), else `heading` (the optical axis). This
  is the reference the engine's world alignment assumes (`compassReference: 'auto'`). Passing the
  optical-axis `heading` unchanged makes a landscape capture (yard: device top horizontal,
  pointing left) read 90° off whenever the compass decides the yaw (e.g. `--no-gps`).
* FramePump metadata: `index` (0-based delivered-frame counter), `segment: 0`, `timeSource:
  'captureTime'`, `imuCalibrationVersion` (0, or −1 without IMU) and `gravityDevice`. The
  gravity is the unit DOWN vector from `accelG − accel` of the newest IMU sample.

`EngineOptions`: `mode` (object for object datasets, scene otherwise), `quality`, `persist: false`,
`preferGPU: true`, and `hfovPriorDeg`. The prior defaults to 67.3°, which is what CameraCapture
reports for an iPhone main camera; the datasets use 65°. `--hfov gt` uses the exact value.
`--mapping`, or `--set engineConfig.<key>=<json>`, sends the `src/engine` `EngineConfig` as the
`engineConfig` field of the init options.

The harness waits for every `LiveUpdate` before it sends the next frame, so there is one frame in
flight. Then it runs `finishCapture`, `refine`, `densify`, `mesh` and the exports, collecting the
progress events. A failing step is recorded in `errors`, the steps that depend on it are skipped,
and a partial result is still written. On `--timeout-min` the runner calls `harnessAbort()`, which
cancels the long operation, and saves what was done.

`EngineClient` timeouts are raised for SwiftShader: init 600 s, frame 300 s, long operations
30 min of inactivity. Set them with `--set timeouts.frameMs=…`.

### Output (`e2e/out/e2e/<dataset>[-tag]/`)

| File | Content |
|---|---|
| `result.json` | Harness result plus the `metrics.mjs` input fields. See below. |
| `sparse.ply`, `dense.ply`, `mesh.ply` | Streamed from the page through `window.__e2eWrite`. |
| `metrics.json` | `computeMetrics` output, sparse-cloud accuracy, live-trajectory metrics and threshold checks. |

The harness fields of `result.json` are:

* `config` (resolved), `dataset`, `engine` (init result, respawns, status changes).
* `timings`: harness steps in ms. `stageTimings`: per engine progress stage, with the
  `src/app/stages.ts` UI stage and its duration.
* `frames`: one record per delivered frame, with the decision, guidance, tracking, counters,
  overlay count, engine `timings`, wall and decode ms, and any error.
* `decisions`, `guidance` and `tracking`: counts.
* `live`: keyframes, registered images, points, `timeToInit`, frame and keyframe ms stats, and the
  last live cameras.
* `progress`: decimated progress events.
* `sparse`: intrinsics normalised to the dataset size, mapped `images`, `metric`, `stats`,
  `pointsCount`, and the `unmapped` / `duplicates` camera ids.
* `dense`, `mesh` and `exports` (sizes only), plus `logs`, `errors` and page `memory`.

The `metrics.mjs` fields are `intrinsics`, `images`, `metric`, `numInputImages` and `pointsFile`.
To re-score a run, use `node e2e/metrics.mjs e2e/data/object e2e/out/e2e/object/result.json`.

**SparseCamera → dataset frame.** Each camera is mapped by its `frameIndex` (the `FrameInput.index`
it came from, an extension field of the engine's `SparseCameraEx`), then by its `t` (nearest
delivered frame within 1 ms), then by its `id` read as the keyframe ordinal. The ordinal is the
k-th frame answered with `'keyframe'`. `sparse.images[].via` records which rule was used.
Engine intrinsics at another resolution are rescaled with the §2 rule. Transposed sizes are
flagged in `sparse.intrinsicsNote`.

### Checks and thresholds (`e2e/thresholds.json`)

Limits merge in order: `default`, then `<meta.scene>`, then `datasets[<name>]`. The check names
are defined in `e2e/lib/e2e-checks.mjs` (`CHECKS`). Initial values:

| check | object | yard |
|---|---|---|
| `registrationRate`: registered / images tried (`stats.images`, else live keyframes) | ≥ 0.9 | ≥ 0.9 |
| `ateRelative`: Sim3 ATE / scene extent | ≤ 1 % | ≤ 2 % |
| `focalErrorPct` (absolute) | ≤ 2 % | ≤ 2 % |
| `scaleError` (absolute; only when the result claims metric) | ≤ 10 % | ≤ 10 % |
| `gravityErrorDeg` / `headingErrorDeg` (only when claimed) | ≤ 2° / ≤ 10° | ≤ 2° / ≤ 10° |
| `denseAccuracyMedianRel` / `denseAccuracyMedian` | ≤ 1 % of extent | ≤ 0.1 m |
| `denseCompleteness` at τ = `meta.denseTau` | ≥ 0.5 | ≥ 0.2 |
| `claims`: the result must claim these when the sensors were passed | metric, gravity | metric, gravity, north |

The yard dense limits are lower because even with exact poses the dense stage reaches only
0.23–0.34 completeness on that scene (see the reference run above). A check that applies but
has no value fails; for example, a missing trajectory fails `ateRelative`. Dense checks are
skipped with `--no-dense`.

The exit code is 0 when every pipeline finished without errors and every enforced check passed,
1 otherwise, and 2 for usage errors. Mock runs report the checks but do not enforce them unless
`--enforce` is given; `--no-thresholds` stops enforcing them for any engine.

`metrics.json` also holds two informational blocks:

* `sparseCloud`: accuracy of `sparse.ply`, with the same definition as the dense accuracy.
* `live`: ATE and rotation error of the last live keyframe poses. They are mapped by keyframe
  order, which assumes that `LiveUpdate.cameras` lists every keyframe in capture order.

### By hand

Open `harness.html?dataset=object&engine=mock&maxFrames=40&densify=1` on the dev server
(`npm run dev`). It runs on load and shows the result. The query keys are `engine`, `quality`, `mode`, `stride`,
`maxFrames`, `start`, `fps`, `imu`, `gps`, `heading`, `densify`, `mesh`, `exports` and `mockSpeed`.
From the console, `await runPipeline({...})` returns the same `HarnessResult`. When `window.__e2eWrite`
is missing, the dense cloud is embedded as base64 in `densePoints`.

### Engine end-to-end results (2026-10, SwiftShader)

```sh
node e2e/run-e2e.mjs object --dense-size 480 --tag final --set 'engineConfig={"featureBackend":"cpu"}'
node e2e/run-e2e.mjs yard   --dense-size 480 --tag final --set 'engineConfig={"featureBackend":"cpu"}'
```

Real worker through `EngineClient`, every frame, IMU + compass (+ GPS for yard), fast preset with
the tuned `DENSE_PRESETS` (4 photometric + 2 geometric iterations, 5 sources, 40 views), dense
images at 480 px instead of 800 px. `featureBackend: 'cpu'` runs SIFT / matching on the CPU (same
features as the GPU kernels; GPU SIFT on SwiftShader costs 4–8 s per frame), PatchMatch runs on
the GPU. Every check of `thresholds.json` passes.

| | object (200 frames → 90 keyframes) | yard (240 frames → 80 keyframes) |
|---|---|---|
| registered / keyframes (live at capture end) | 90 / 90 (90) | 80 / 80 (80) |
| ATE RMSE (Sim3) / relative | 0.29 mm / 2.1e-4 | 8.1 mm / 2.1e-4 |
| rotation error mean / max | 0.054° / 0.094° | 0.058° / 0.093° |
| focal error / principal point error | +0.030 % / 0.46 px | +0.065 % / 0.31 px |
| scale error (source) | +0.37 % (IMU, σ 2 %) | +0.013 % (IMU + GPS) |
| gravity / heading error (yaw source) | 0.013° / −0.36° (compass) | 0.013° / −2.66° (GPS) |
| camera–IMU time offset (true −25 ms) | −25.0 ms | −24.9 ms |
| sparse cloud accuracy p50 / p90 | 0.58 / 3.3 mm | 15 / 79 mm |
| dense accuracy p50 / p90 | 0.24 / 0.81 mm | 6.7 / 42 mm |
| dense precision / completeness / F (τ) | 0.9997 / 0.996 / 0.998 (13.6 mm) | 0.964 / 0.379 / 0.545 (0.1 m) |
| dense points / mesh triangles | 1.31 M / 499 k | 1.58 M / 662 k |

Wall time per step (s; 4-core VM shared with other SwiftShader jobs, load 8–14): object — capture
119 (frame p50 0.51 s), finishCapture 0.8, refine 81 (matching 45, reconstruction 36, alignment 0.4),
depth maps 1780, fusion 4, mesh 5, exports 1; yard — capture 114 (p50 0.40 s), finishCapture 0.7,
refine 56 (41 / 14 / 0.1), depth maps 1261, fusion 5, mesh 5, exports 1. Depth maps cost 12–45 s per
480 px view on SwiftShader; this says nothing about Apple GPUs.

Other runs: `--engine inline` (object, sparse) gives the same accuracy (ATE 2.0e-4, rotation 0.054°,
scale +0.34 %). Yard with `--no-gps` (compass yaw): heading −0.15°, scale +0.30 % (IMU only). The app's
GPU path (default `featureBackend`, WebGPU SIFT + matcher in the worker; object `--stride 2 --no-dense`):
88/88 registered, ATE 2.0e-4, rotation 0.039°, focal +0.06 %, scale +0.86 %, gravity 0.035°, heading
−0.99°, principal point 0.20 px — the same accuracy class, but 25 s per frame and 1.5 s per refine pair
on SwiftShader (capture 38 min, refine 24 min).
