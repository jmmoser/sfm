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
