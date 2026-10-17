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
node e2e/generate-synth.mjs [object|yard|all|object-real|yard-real|real] [--frames N] [--fps F]
     [--size WxH] [--duration S] [--out DIR] [--name NAME] [--seed N] [--blur N] [--no-noise]
     [--depth-every N] [--gt-points N] [--gt-focus-points N] [--time-offset MS]
     [--texture-scale X] [--visibility-every N] [--supersample N]
     [--rs-readout MS] [--rs-bands N] [--focus-breathing AMP] [--wb-drift AMP]
     [--exposure-jitter AMP] [--jpeg-quality Q] [--adaptive-blur PX] [--set key=json ...]

node e2e/generate-synth.mjs all --frames 40     # quick: 40 frames spread over each trajectory
node e2e/generate-synth.mjs yard                # full default: 240 frames (6 fps × 40 s)
node e2e/generate-synth.mjs real                # object-real + yard-real (iPhone capture artefacts)
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
| `frames.json` | `[{index, file, t, tTrue, pose: {R: number[9], t: number[3]}, center, gain, depthFile?}]`; with real-capture artefacts also `f` (GT focal of the frame), `focusDistance`, `wb`, `omega`, `blurPx`, `rsPx`, `blurSamples`, `rsBands`, `renders` (see below) |
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

### Real-capture artefacts (`object-real`, `yard-real`)

iPhone video artefacts the clean datasets lack. Every option is deterministic and OFF by default
(clean datasets are unchanged: same JPEGs, poses and IMU; only `meta.config` / `meta.render` list
the new keys). The presets `object-real` (portrait θ = 0, 720×960, 8 fps × 25 s, 200 frames) and
`yard-real` (landscape θ = 90, 960×720, 6 fps × 40 s, 240 frames) are the clean scenes with all
of them on (`SYNTH_PRESETS` in `src/harness/synth/config.ts`); options and `--set` override a
preset, e.g. an ablation without rolling shutter:
`node e2e/generate-synth.mjs object-real --name object-real-gs --set rsReadoutMs=0`.

| Artefact | Config (preset value) | Model |
|---|---|---|
| Rolling shutter | `rsReadoutMs` (22), `rsBands` (48 max), `rsBandStepPx` (0.3), `rsReverse` (false); `--rs-readout`, `--rs-bands` | see below |
| Focus breathing | `focusBreathing` ({amplitude 0.006, lagS 0.25, …}; `--focus-breathing AMP`) | per-frame f_i = f0·(1 + b_i): AF target = median z-depth of the central ±15 % of the image at the GT pose; first-order AF lag (0.25 s) in diopters; b linear in diopters, scaled to span exactly ±amplitude over the sequence (near → longer focal). `meta.intrinsics.f` = f0, `frames.json` `f` = f_i. k1, k2, cx, cy fixed |
| Hand shake | `trajectory.shake` ({tremorDeg 0.06, tremorM 1e-4, pans}) | tremor: 3 sinusoids per axis at 8–12 Hz, ±40 % amplitude modulation (yaw / pitch / roll and position); fast "glance" pans: Gaussian out-and-back yaw / pitch bumps, own peak rate 0.607·A/σ = 1.05–1.4 rad/s (object 3 at 5.5 / 12.8 / 19.6 s, yard 4 at 6 / 15 / 24.5 / 33 s; body rate peaks ≈ 1.4–1.5 rad/s with the trajectory's own motion). Exact jets, so the IMU (60 Hz, noise as before) is exact |
| Motion blur | `effects.motionBlurPxPerSample` (1.5), `effects.motionBlurSamples` (12 max); `--adaptive-blur PX` | per band: 1 + ⌈image motion over the exposure / 1.5 px⌉ sub-frames (1 below 0.25 px) at the midpoints of equal exposure slices; exposure 12 ms object, 4 ms yard |
| Exposure / WB drift | `exposureJitter` (0.15), `whiteBalanceDrift` (0.03); `--exposure-jitter`, `--wb-drift` | ±15 % slow auto-exposure (7 s and 2.9 s periods, ±1.5 % per-frame flicker); WB gains r = 1 + A·w, b = 1 − A·w (13 s and 4.1 s), g = 1 + 0.3·A·sin(9.7 s), in linear light |
| JPEG | `jpegQuality` (0.85); `--jpeg-quality` | |

**Rolling shutter: direction, sign and timestamp.** The iPhone sensor buffer is sensor-landscape,
upright for θ = 90 (docs/research/ios-camera.md §1.4), and is read from its top row (row 0) to
its bottom row. In the device frame that sweep runs from the device +x (right-edge) side to −x
whatever the UI orientation, so with a = `R_cam_device`(θ)·x̂_dev:

* θ = 0 (portrait, `object-real`): the readout sweeps image **columns from the right edge
  (u = W−1) to the left (u = 0)**;
* θ = 90 (landscape, `yard-real`): image **rows from the top (v = 0) to the bottom (v = H−1)**;
* θ = 180: columns left → right; θ = 270: rows bottom → top. `rsReverse` flips the sweep.

Line i (0-based, in readout order, of L = W or H lines) is exposed around
`t_frame + (i/(L−1) − 0.5)·rsReadoutMs` (line delay rsReadoutMs/(L−1); first line at −11 ms, last
at +11 ms for 22 ms). **`tTrue`, `t` = tTrue + timeOffsetMs, and the GT pose of a frame are those
of mid-readout** (s = 0.5, also the middle of the middle line's exposure); the IMU stays on the
true clock. Rendering: the lines are split into N contiguous bands; each band is rendered with
the poses at `t_frame + (s̄ − 0.5)·readout` (s̄ = mean readout coordinate of its lines, ± the
blur sub-frame offsets) into exactly the part of the undistorted target its distorted pixels
sample, then resampled through the inverse RADIAL distortion under a scissor, so the lens
distortion is applied consistently per line. N adapts per frame: N = ⌈rsPx / max(0.3 px,
0.1·blurPx)⌉ in [4, 48] (rsPx = image motion over the readout), bounding the step between bands.
GPU tests (`realism.gpu.test.ts`, portrait column bands and landscape row bands) check that bands
with one pose (or several identical blur sub-frames) reproduce the global-shutter image exactly
and that markers seen during a 1.3 rad/s rotation land within 0.07 px of their band pose's
projection (≥ 1.2 px away from the mid-readout projection).
Depth maps and visibility stay global-shutter at the GT (mid-readout) pose.

Modelling assumptions (docs/research/ios-camera.md establishes only the sensor-landscape buffer,
upright at θ = 90; the readout order and the timestamp's reference instant are not documented
there): native row 0 is read first, and the frame timestamp is mid-readout. If iOS stamps the
start of readout or of exposure instead, the difference is a constant (≈ readout/2, plus
exposure/2) that a camera–IMU time-offset estimate absorbs like the datasets' +25 ms;
`rsReverse` covers a bottom-to-top sensor.

Ablation datasets (one artefact family on, everything else at the clean values) are built from
the presets with `--set`; the recipes of the baselines below:

```sh
CLEAN='--set exposureJitter=0.05 --set jpegQuality=0.92 --set whiteBalanceDrift=0'
NOBLUR='--set effects.motionBlurPxPerSample=0 --set effects.motionBlurSamples=1'
# rolling shutter only (smooth trajectory, no blur)
node e2e/generate-synth.mjs object-real --name abl-object-rs $CLEAN $NOBLUR --set focusBreathing=null --set trajectory.shake=null
node e2e/generate-synth.mjs yard-real   --name abl-yard-rs   $CLEAN $NOBLUR --set focusBreathing=null --set trajectory.shake=null
# hand shake + fast pans with their adaptive motion blur (global shutter)
node e2e/generate-synth.mjs object-real --name abl-object-shake $CLEAN --set rsReadoutMs=0 --set focusBreathing=null
node e2e/generate-synth.mjs yard-real   --name abl-yard-shake   $CLEAN --set rsReadoutMs=0 --set focusBreathing=null
# focus breathing only
node e2e/generate-synth.mjs object-real --name abl-object-breath $CLEAN $NOBLUR --set rsReadoutMs=0 --set trajectory.shake=null
```

`meta.captureArtefacts` records the settings and statistics (`rollingShutter`, `focusBreathing`
incl. fMin / fMax / fMean, `shake` incl. `panPeakRates`, `angularSpeed` of the trajectory, and
per-frame p50 / p90 / max of `omega`, `blurPx`, `rsPx`, `renders`, `rsBands`, `focusDistance`);
it is `null` for clean datasets. Per frame, `frames.json` adds `f`, `focusDistance`, `wb`, `omega`
(|ω| at the frame time, rad/s), `blurPx` / `rsPx` (largest image motion over the exposure / the
readout at the AF depth), `blurSamples`, `rsBands` and `renders`. `metrics.mjs` additionally
reports the per-frame GT focal range and the focal error against the mean GT focal of the
registered frames (`intrinsics.perFrameFocal`, informational; `focalErrorPct` stays against f0).
Generation is slower: about 3.5 s per frame on SwiftShader (≈ 70 ms per band render).

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
node e2e/run-dense-gt.mjs [object|yard|all|<dataset> ...] [--max-size 480] [--views N] [--stride N] [--first N]
     [--frames i,j,...] [--preset fast|balanced|max] [--sources 8] [--iterations 6] [--geom-iterations 3]
     [--geom-passes 1] [--no-geometric] [--multiscale] [--tsdf] [--debug-views N] [--sparse N]
     [--save-maps] [--load-maps RUN] [--fusion-input geo|photo] [--oracle] [--tag NAME] [--out DIR]
     [--set key=json ...] [--no-render] [--no-metrics] [--no-consistency] [--verbose]

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
`fusion.haloFilter={}`), `preHalo` (`removeDepthHalos` on the fusion input, as the engine does;
`true` or an options object, e.g. `--set 'preHalo={"skyMaxStd":0}'` turns the sky test off),
`haloSkyUp=true` (the halo filter gets each view's GT up direction as `skyUp`, as an engine with a
gravity-aligned model would pass it), `haloRotate=k` (`preHalo` on views rotated by k clockwise
quarter turns, result rotated back: frames that are not upright, `src/harness/dense-gt/halo-rotation.ts`),
`filter` (`filterPointCloud` on the fused cloud), `geometricPasses`, `multiScale`,
`textureAnalysis={}` (per texture threshold: GT error classes of the fusion-input and fused pixels
an untextured / sky gate would remove, plus the coherence of the top-connected untextured
regions; `src/harness/dense-gt/texture-analysis.ts`). When a halo filter is configured the run
also prints which GT error classes it rejects.

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
* for object, precision inside and outside the focus box;
* sky-coloured points (light / saturated blue) and their share of the points farther than τ
  (the yard halo fringe);
* whole scene: precision and completeness on the base GT set without the ROI (object: includes
  the room walls, which the focus-set metrics do not see).

Reference run (2026-10): 480 px; 24 views for object and 30 for yard; 8 sources; 6 photometric
and 3 geometric iterations; default fusion; τ = `denseTau`:

| scene | precision | completeness | F | same with `fusion.minViews=2` | oracle (GT depth), minViews 3 / 2 |
|---|---|---|---|---|---|
| object | 0.999 | 0.960 | 0.979 | P 0.998, C 0.988, F 0.993 | C 0.983 / 0.998 |
| yard | 0.989 | 0.231 | 0.374 | P 0.978, C 0.338, F 0.502 | C 0.316 / 0.461 |

Tuned (`--preset balanced`, same views and 480 px: view-selection near margin 0.5, fusion
`minViews` 2, normal 15°, `removeDepthHalos` on the final maps; see ARCHITECTURE §6 "Dense tuning
results" for every experiment): object P 0.9997, C 0.987, F 0.994 (mesh F 0.963 → 0.987); yard
P 0.981, C 0.352, F 0.518 (mesh F 0.348 → 0.492). With the sky test of `removeDepthHalos`
(default since 2026-10, ARCHITECTURE §6 "Sky / low-texture halo removal"; yard maps saved under
`yard-skyb`): yard P 0.9854, C 0.3521, F 0.5188, sky-coloured points farther than τ 5,722 → 707,
mesh F 0.492 → 0.495; object unchanged (P 0.9997, C 0.9874, mesh F 0.9873 → 0.9868).

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
  North reference (`--set headingNorth='"magnetic"|"app"|"true"'`; the datasets store TRUE north):
  by default (dataset with a location, GPS fed) the headings are converted to MAGNETIC with
  WMM2025 at `meta.originLatLonAlt` and the run's date and flagged `north: 'magnetic'`, so the
  engine has to apply the declination at its first GPS fix (yard: ≈ +15° E). `app` = magnetic
  until a fix is delivered, then true north via `CompassSensor.setLocationFromFix` (what the app
  sends); `true` = unchanged (default for object, which has no location, and with `--no-gps`).
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

**Diagnostics** (off by default; large results):

* `--set liveTrace=true` (any engine): `result.liveTrace` keeps the camera poses of every
  `LiveUpdate` that carries the model (frame index, keyframe / registered / point counts). Poses
  are the registered keyframes sorted by id, so they map to `live.keyframeIndices` only while every
  keyframe is registered.
* `--engine inline --set debugDump=true --set 'engineConfig={"mapping":"inline",…}'`:
  `result.debug` (`src/harness/engine/debug-dump.ts`, reads engine internals) has, after every
  keyframe frame, each keyframe's live registration state, 3D-point count and pose
  (`keyframeTrace`), and after finishCapture and after refine the live model plus every verified
  pair (keyframe ids, frame indices, model, inliers, median triangulation angle, R, t). Comparing
  R / t with the GT relative pose separates wrong pairs from wrong registrations.

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
`maxFrames`, `start`, `fps`, `imu`, `gps`, `heading`, `headingNorth`, `densify`, `mesh`, `exports` and `mockSpeed`.
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

True-north heading (2026-10, `--no-dense`, after the WMM2025 / compass ⊕ GPS yaw change): the yard compass is replayed
as MAGNETIC headings (declination 14.87° at the dataset location) and converted back by the engine at its first GPS
fix; GPS and compass yaws are fused (compass weight 0.80): heading error −0.64° (was −2.66° with the GPS yaw alone,
reproduced with `--no-heading`); `--set headingNorth='"app"'` gives the same −0.64°. Object (no location ⇒ true-north
replay): −0.36°, unchanged. Yard `--no-gps --set headingNorth='"magnetic"'`: +14.7° = magnetic north, `northAligned`
false (the north claim fails, as it should without any location).

Other runs: `--engine inline` (object, sparse) gives the same accuracy (ATE 2.0e-4, rotation 0.054°,
scale +0.34 %). Yard with `--no-gps` (compass yaw): heading −0.15°, scale +0.30 % (IMU only). The app's
GPU path (default `featureBackend`, WebGPU SIFT + matcher in the worker; object `--stride 2 --no-dense`):
88/88 registered, ATE 2.0e-4, rotation 0.039°, focal +0.06 %, scale +0.86 %, gravity 0.035°, heading
−0.99°, principal point 0.20 px — the same accuracy class, but 25 s per frame and 1.5 s per refine pair
on SwiftShader (capture 38 min, refine 24 min).

### Real-capture baselines (2026-10, sparse, CPU features, worker engine)

`node e2e/run-e2e.mjs <dataset> --no-dense --no-mesh --tag sparse --set 'engineConfig={"featureBackend":"cpu"}'`
(quality fast, all sensors; outputs `e2e/out/e2e/<dataset>-sparse`; the clean references are
`object-clean-sparse2` / `yard-clean-sparse2`). Scale error = 1/s − 1 of the Sim3; gravity /
heading in the result's own world frame; Δt = the engine's camera–IMU offset (truth −25 ms);
decisions = skip / keyframe / reject_blur / reject_motion; lost = frames reported `lost`.

| dataset | kf reg (model) | ATE / extent | rot° p50 | focal % (vs mean GT f) | pp px | scale % (source) | grav° | head° | Δt ms | decisions | lost | sparse P |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| object (clean) | 90/90 (live) | 2.1e-4 | 0.05 | +0.03 | 0.4 | +0.3 (imu) | 0.01 | −0.36 | −25.0 | 110/90/0/0 | 0 | 0.981 |
| object-real | 89/89 (live) | 8.6e-4 | 0.73 | −0.04 (−0.14) | 8.1 | −4.3 (imu) | 0.03 | −0.73 | −25.2 | 100/89/11/0 | 0 | 0.977 |
| object-real, `refinePrincipalPoint: false` | 89/89 (live) | 8.6e-4 | 0.14 | −0.59 (−0.69) | 1.8 | −4.3 (imu) | 0.02 | −0.75 | −25.2 | 100/89/11/0 | 0 | 0.978 |
| abl-object-rs | 91/91 (live) | 5.1e-4 | 0.49 | −0.40 | 5.0 | −3.0 (imu) | 0.01 | −0.67 | −24.5 | 109/91/0/0 | 0 | 0.978 |
| abl-object-shake | 88/88 (live) | 1.8e-4 | 0.04 | +0.02 | 0.3 | −0.0 (imu) | 0.03 | −0.51 | −25.1 | 99/88/13/0 | 0 | 0.979 |
| abl-object-breath | 90/90 (live) | 8.0e-4 | 0.47 | +0.32 (+0.48) | 5.9 | −0.1 (imu) | 0.01 | −0.61 | −25.0 | 110/90/0/0 | 0 | 0.979 |
| yard (clean) | 80/80 (live) | 2.2e-4 | 0.05 | +0.07 | 0.3 | −0.0 (imu+gps) | 0.01 | −0.64 | −24.9 | 160/80/0/0 | 0 | 0.919 |
| **yard-real** | 75/75 (live) | **0.103** | **152** | −0.08 (−0.19) | 6.1 | −1.9 (gps) | **3.55** | +7.80 | −29.4 | 152/75/12/1 | 22 | **0.028** |
| abl-yard-rs | 80/80 (live) | 7.8e-4 | 0.25 | +0.05 | 2.5 | +5.3 (imu+gps) | 0.02 | −0.95 | −27.8 | 160/80/0/0 | 0 | 0.860 |
| **abl-yard-shake** | 76/76 (offline) | **0.033** | 1.21 | +0.01 | 0.2 | −4.7 (gps) | 0.71 | −0.22 | −25.0 | 150/76/13/1 | 26 | **0.256** |

object-real with dense (`--dense-size 480`): P 0.9991, C 0.9926, F 0.9958 (accuracy p50 1.3 mm).
Every object run passes. Rolling shutter and focus breathing are absorbed by the free principal
point (2.5–8 px off, ≈ a constant 0.25–0.7° camera rotation; without principal-point refinement the
rotation error is 0.14° but the focal −0.6 %), and rolling shutter biases the IMU scale (−3 / −4 %
object, +5 % yard, within 1–2 σ of the reported uncertainty) and the time offset (−28 / −29 ms).

**Yard breakdown (shake, not rolling shutter: `abl-yard-rs` passes).** Replayed with
`--engine inline --set debugDump=true` and inline mapping (same failure, see Diagnostics above):

1. Pan 2 (t = 15 s, 1.1–1.3 rad/s): frames 91–94 are rejected as blurry (< 0.6 × the median
   sharpness) after 89 / 90 were deferred "waiting for a sharper frame", so the keyframe gap is
   87 → 95 (34.5°, `abl-yard-shake`) / 89 → 95 (45°, `yard-real`). The first keyframe after the
   gap sees a thin map (`yard-real`: 95 registers with 122 point observations, ≈ 700 is usual);
   the next ones see 40, 20, 21, 0, 0 … 3D points and stay pending although their mutual pairs
   have 200–400 inliers: 8–10 keyframes unregistered, `tracking_lost` for 3.7–4.5 s after the pan.
2. Relocalisation through false pairs: the spatial candidates come from the last REGISTERED
   keyframe (stale anchor 87 / 95 → its neighbours 79 / 81 / 84 / 85, which look ≈ 85–100° away).
   Both views see the west fence ≈ 20 m apart; its repeated boards verify as `calibrated` pairs
   with 59–79 inliers, a 131–137° median triangulation angle and R 127–136° off the truth. Keyframe
   127 registers through them 74° / 92° off, and `retryPending` then registers 124 … 95 in the same
   wrong frame: one model made of two rigid fragments (each with 1–8 cm ATE) joined by a
   55–90° / ≈ 10 m error.
3. Refine compounds it: its retrieval / spatial pairs (spatial from the broken live model) add
   ≈ 70 more wrong pairs (R > 10° off, many with 50–150 inliers); the offline model then misplaces keyframes too (`abl-yard-shake`: only 87, by
   53°, so the offline model wins; `yard-real`: keyframe 85 and the fragment 95 … 184 rotated by
   ≈ 160°, 188 … 239 attached to the start of the loop).
   Live vs offline is chosen by registered count, then mean reprojection error (0.512 vs 0.517 px).
4. World alignment fits the broken trajectory: the IMU scale is rejected (R² 0.01–0.02; the R² ignores
   the robust weights) → GPS scale (within a fragment the true scale error is −15 % for `yard-real`,
   −7 % for `abl-yard-shake`); `yard-real` gravity 3.5°, heading 7.8°.

With the keyframe gates relaxed (`--set 'engineConfig={"featureBackend":"cpu","keyframe":{"maxDeferMs":0,"blurRatio":0.3}}'`,
only ω > 1.2 rad/s rejected; tags `kfgate`) `abl-yard-shake` (inline engine) is as good as the clean yard: 83/83 keyframes, max gap
33.7°, no `lost` frame, ATE 2.3e-4, rotation 0.04°, scale −0.01 % (imu+gps), gravity 0.04°, heading
−0.69°, sparse P 0.921 — although 15 (capture) / 109 (refine) pairs are still wrong: false pairs are
harmless while registration is continuous. `yard-real` (worker) with the same gates: no `lost` frame and a
live model without a break (live ATE 5.6e-3, all consecutive relative rotations < 2°), but the
offline model, chosen by 0.592 vs 0.611 px, misregisters keyframe 212 by 109° / 14 m, which alone
gives ATE 4.2e-2 (1.1e-3 = 4.2 cm without it), rejects the IMU scale (R² −0.00) and leaves the
GPS scale (−6 %). Likewise the `abl-yard-shake` baseline without its keyframe 87: ATE 2.5e-4, GPS
scale −7.3 %.

### Real-capture robustness (2026-10, CPU features, worker engine)

What changed (details and the evidence for each choice: ARCHITECTURE §6 "Real-capture robustness"):
hard rotation limit (keyframes forced through fast pans) and best-of-window keyframing; gyro gates on
verified pairs, registrations and initial pairs (live and refine) and gyro-consistent model choice
(breaks, isolated outliers, explained correspondences); spatial candidates from the gyro-propagated pose;
gyro bias / camera–IMU offset calibration in refine; robust R² of the IMU scale; rolling-shutter estimation
(short subset BA per hypothesis, gyro rotation + translation from the neighbouring keyframes × point depth,
2 passes) with keypoint correction and rolling-shutter-corrected dense views; per-image focal lengths (focus
breathing) with the shared camera re-refined without the breathing.
Same command as the baselines above (`--no-dense --no-mesh`, `featureBackend: 'cpu'`), final code, tag `final4`
(the -real rows: the sparse part of the 480 px dense runs `final4-dense480`); the baselines in parentheses.

| dataset | kf reg (model) | ATE / extent | rot° p50 | focal % (vs mean GT f) | pp px | scale % (source) | grav° | head° | Δt ms | decisions | lost | sparse P | rolling shutter (truth) | per-image focal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| object (clean) | 92/92 (live) | 2.1e-4 (2.1e-4) | 0.07 (0.05) | +0.04 | 0.7 (0.4) | +0.3 (imu) | 0.01 | −0.83 (−0.36) | −24.9 | 108/92/0/0 | 0 | 0.980 | none | kept shared |
| object-real | 89/89 (live) | **2.8e-4** (8.6e-4) | **0.08** (0.73) | +0.08 (−0.04) | **1.0** (8.1) | **−0.8** (−4.3) (imu) | 0.02 | −0.54 | −25.1 | 102/89/9/0 | 0 | 0.980 | cols −21.7 ms (−22) | refined, r = 0.92 |
| yard (clean) | 80/80 (live) | 2.2e-4 (2.2e-4) | 0.05 (0.05) | +0.07 | 0.4 (0.3) | +0.1 (imu+gps) | 0.01 | −0.64 | −24.9 | 160/80/0/0 | 0 | 0.920 | none | kept shared |
| yard-real | 81/81 (live) | **3.1e-4** (0.103) | **0.09** (152) | +0.15 (−0.08) | **0.3** (6.1) | **+1.6** (−1.9) (imu+gps) | **0.06** (3.55) | **−0.93** (+7.80) | **−25.1** (−29.4) | 154/81/5/0 | **0** (22) | **0.899** (0.028) | rows 22.0 ms (22) | refined, r = 0.87 |

Bold = changed materially against the realism engineer's baseline (in parentheses). r = correlation of the
per-image focal lengths with the GT focal lengths (rms error 0.07 / 0.13 % against a GT spread of 0.19 / 0.26 %;
a shared focal has an rms error equal to the spread). The clean datasets keep their ATE; their rotation p50
0.05 → 0.07° (object) is the orbit's weakly observable pitch ↔ c_y mode (pp 0.4 → 0.7 px) moving with the pair
set (the gyro gate drops 12 false object pairs: `p0a` 0.07° with identical keyframes), the heading change follows
the 2 extra keyframes (best-of-window: `nobow` −0.36°).

Ablations (one artefact family each; final code, tags `final4` / `rst2`):

| dataset | ATE / extent | rot° p50 | pp px | scale % | Δt ms | rolling shutter (truth) | per-image focal |
|---|---|---|---|---|---|---|---|
| abl-yard-shake | 1.6e-4 (0.033) | 0.05 (1.21) | 0.3 (0.2) | −0.2 (−4.7, gps) | −25.1 (−25.0) | none | kept shared |
| abl-yard-rs | 2.5e-4 (7.8e-4) | 0.05 (0.25) | 0.5 (2.5) | +0.7 (+5.3) | −24.9 (−27.8) | rows 22.4 ms (22) | kept shared |
| abl-object-rs | 2.2e-4 (5.1e-4) | 0.04 (0.49) | 0.3 (5.0) | +0.0 (−3.0) | −25.5 (−24.5) | cols −19.1 ms (−22) | kept shared |
| abl-object-breath | 2.7e-4 (8.0e-4) | 0.08 (0.47) | 1.0 (5.9) | +0.2 (−0.1) | −25.0 (−25.0) | none | refined, r = 0.98 |

Step by step on the -real datasets (each row adds to the previous; tags `p0a`, `p1c`, `calib2` / `p2a`, `fin`,
`expl`, `final3-dense480`, `rst2`):

| step | yard-real ATE / rot° / pp px / scale % | object-real ATE / rot° / pp px / scale % |
|---|---|---|
| baseline | 0.103 / 152 / 6.1 / −1.9 (gps) | 8.6e-4 / 0.73 / 8.1 / −4.3 |
| + hard rotation limit, gyro gates, gyro model choice, spatial candidates, live calibration (`p0a`) | 2.8e-3 / 0.43 / 6.3 / +3.9 | 1.1e-3 / 0.75 / 8.6 / −3.9 |
| + rolling shutter (rotation only, subset BA), best-of-window (`p1c`) | 1.2e-3 / 0.15 / 0.7 / +2.0 | 7.6e-4 / 0.45 / 3.1 / −1.1 |
| + per-image focal with shared re-refinement (`calib2` / `p2a`) | 7.0e-4 / 0.12 / 0.7 / +1.1 | 5.4e-4 / 0.38 / 2.2 / −0.2 |
| live calibration off, calibration in refine (`fin`) | 3.4e-4 / 0.10 / 0.5 / +2.3 | 5.5e-4 / 0.41 / 2.5 / −0.6 |
| model choice with explained correspondences (`expl`) | 2.6e-4 / 0.09 / 0.6 / +1.4 (offline) | 5.4e-4 / 0.41 / 2.5 / −0.6 |
| up to 6 rounds of the per-image focal / shared camera alternation (`final3-dense480`) | 3.4e-4 / 0.10 / 0.5 / +2.3 (live) | 4.7e-4 / 0.27 / 0.5 / −0.6 |
| + translational rolling-shutter term, 2 passes (`rst2`; final) | 3.3e-4 / 0.10 / 0.4 / +1.8 (live) | 2.9e-4 / 0.07 / 0.9 / −0.8 |

yard-real: from `final3` on, the live model closes the loop by itself (86.2 % explained, offline 87.1 %, below the
1.5-point gap) and is kept; an unclosed live model measured 2.5e-3 (`expl` chose the offline one, 2.6e-4).

Dense (480 px, 40 views, same preset as the baselines):

| dataset (run) | precision | completeness | F | accuracy p50 / p90 |
|---|---|---|---|---|
| object-real baseline (`dense480`) | 0.9991 | 0.9926 | 0.9958 | 1.3 / 2.4 mm |
| object-real, rotation-only rolling shutter (`final3-dense480`) | 0.9995 | 0.9887 | 0.9940 | 3.1 / 4.0 mm |
| object-real final (`final4-dense480`) | 0.9996 | 0.9933 | 0.9964 | **0.31 / 1.05 mm** |
| object clean (above) | 0.9997 | 0.996 | 0.998 | 0.24 / 0.81 mm |
| yard-real, rotation-only rolling shutter (`final3-dense480`) | 0.961 | 0.347 | 0.510 | 9.6 / 50 mm |
| yard-real final (`final4-dense480`) | 0.961 | 0.352 | 0.515 | 8.8 / 46 mm |
| yard clean (above) | 0.964 | 0.379 | 0.545 | 6.7 / 42 mm |

The yard-real baseline was not densified (its sparse model is broken). The 3.1 mm of `final3` was a RIGID offset of
the cloud against the cameras (`--dense-align sim3+icp`: 0.35 mm p50, as the baseline's 0.35 mm): every camera
pitched by −0.26° with the object 3 mm lower — the rotation-only correction's under-corrected roll term on the
orbit (ARCHITECTURE "Rolling shutter"); the translational term removes it.

Variants measured and rejected:

* Relaxed keyframe gates alone (`kfgate`): fixes `abl-yard-shake` but not `yard-real` (an offline model with one
  keyframe misplaced by 109° wins on reprojection error) — the hard rotation limit keeps the blur rules for
  normal frames, the gyro gates and model choice handle the rest.
* Rolling shutter scored against FIXED points (motion-only BA): right axis and direction, but rows 3.3 / cols −4.6 ms
  on yard-real / object-real (points triangulated from distorted pans resist); a subset BA with only the 30 fastest
  images free: 17.2 / −8.1 ms; every image free: 21.9 / −20.7 ms. Shared focal fixed / free / free + principal point
  in that BA: −15.1 / −16.8 / −16.9 ms on abl-object-rs.
* Rotation-only rolling shutter (the first P1 version): object-real rotation 0.27° (a constant −0.26° pitch), dense
  p50 3.1 mm, abl-object-rs −16.5 of −22 ms and rotation 0.21°. With the translational term, ONE pass: object-real
  2.98e-4 / 0.11° / pp 1.4 px, abl-object-rs 2.26e-4 / 0.05°; two passes (kept): 2.87e-4 / 0.07° / 0.9 px and
  2.24e-4 / 0.04° (on a synthetic orbit built on distorted keypoints the second pass moves −18.5 → −21.7 of −22 ms).
* Per-image focal accepted on a ≥ 3 % reprojection drop: accepted on object-real while absorbing residual
  rolling shutter (correlation −0.84 with |ω|, lag-1 autocorrelation −0.14, rms error 0.32 % > the 0.19 %
  spread) yet rejected on abl-object-breath where it was right (−1.6 %). Now ≥ 0.5 % AND autocorrelation ≥ 0.6.
  Without the shared re-refinement (forced on abl-object-breath): ATE 4.2e-4, pp 5.6 px; one round 3.7e-4 / 4.6 px;
  2 / 4 / 6 rounds: pp 3.6 / 2.0 / 1.0 px, rotation 0.29 / 0.17 / 0.08°. A 1 % prior instead of 0.5 %: no change.
* Gyro bias / offset calibration DURING capture (now opt-in): yard ATE 2.4–2.7e-4 in 3 runs (2.1–2.2e-4
  without), yard-real 7.0e-4 (3.3e-4), object 2.19e-4 (2.05e-4). Its bias is biased by rolling shutter
  (x 0.5 vs 3.0 mrad/s), and the gyro tolerance tightened to 0.1°/s with it rejected 4 correct 31–35 s
  loop-closure pairs (gyro 5° off GT there) — the tolerance stays 0.3°/s.
* Best-of-window: neutral here (object-real: identical decisions; object: 3 of 92 keyframes one frame
  earlier, ATE 2.19e-4 either way, rotation 0.06 → 0.08°); kept on.

`e2e/thresholds.json` has `datasets` entries for `object-real` / `yard-real`, tighter than their scene's limits so
that what these datasets were built for is caught (checked on the stored runs: the baselines and `p0a` fail them, on
object-real also `expl` and the rotation-only `final3`; `final4` / `rst2` pass with a 2–6× margin, completeness
0.35 vs 0.28 on yard-real): registration ≥ 0.95, ATE / extent
≤ 7e-4 / 1e-3, rotation p50 ≤ 0.25° / 0.3°, focal ≤ 0.5 / 1 %, scale ≤ 2.5 / 5 %, gravity ≤ 0.5°, heading ≤ 3°,
dense accuracy p50 ≤ 6e-4 of the extent (0.8 mm) / 2 cm, completeness ≥ 0.98 / 0.28.

Refine time (SwiftShader host, two runs in parallel): object-real 77 s (baseline 65 s: ≈ 25 BAs of the
rolling-shutter passes, two live-model re-adjustments, the per-image focal rounds), yard-real 65 s (baseline
112 s, broken model), clean object / yard 55 / 48 s (rolling-shutter screen only). The yard-real gyro gates reject
≈ 120 verified pairs per run (live + refine), every one checked against GT ≥ 6° off while the bias-compensated
gyro is within 0.2–1.6°.

#### Adversarial review (2026-10, same command, tag `rev1`; implementer's final runs in parentheses)

| dataset | kf reg (model) | ATE / extent | rot° p50 | pp px | focal % | scale % | live-trajectory ATE / extent |
|---|---|---|---|---|---|---|---|
| object | 92/92 (live) | 2.2e-4 (2.1e-4) | 0.07 (0.07) | 0.7 (0.7) | +0.04 | +0.3 | **1.7e-4** (1.9e-3) |
| yard | 80/80 (live) | 2.2e-4 (2.2e-4) | 0.05 (0.05) | 0.4 (0.4) | +0.07 | +0.1 | 1.3e-3 (1.4e-3) |
| object-real | 89/89 (live) | 2.8e-4 (2.9e-4) | 0.08 (0.07) | 0.9 (0.9) | +0.09 | −0.9 | 1.0e-3 (1.0e-3) |
| yard-real | 81/81 (offline) | 2.1e-4 (3.3e-4, live) | 0.08 (0.10) | 0.3 (0.4) | +0.14 | +1.0 | 4.5e-3 (4.4e-3) |

object-real dense 480 px (`rev3-dense480`): P 0.9995, C 0.9936, accuracy p50 / p90 0.31 / 1.06 mm (`final4`: 0.9996 /
0.9933 / 0.31 / 1.05 mm). The sparse pipeline is unchanged by the fixes (same keyframes, readouts cols −21.6 / rows 21.9 ms); yard-real picked
the offline model this time (live 84.9 vs offline 87.1 % explained: the live model's loop closure varies with the
background-mapping timing). Fixed: the live-trajectory metric mapped a best-of-window keyframe to the LATER frame whose
update reported it (object: 3 held keyframes ⇒ 1.9e-3 instead of 1.7e-4; `keyframeSourceIndex`); the dense colour
thumbnails (fusion / TSDF colours) were the UNcorrected images in session; the dense ω were re-queried after the
alignment moved the time offset (now the ω each keyframe was corrected with, `CaptureState.rsOmega`); COLMAP POINTS2D
after a reload / respawn were the original keypoints against rolling-shutter-corrected poses (up to ≈ 12 px at the
image edges in fast pans); `rsOriginal` / the held frame were missing from the capture memory estimate.

Measured, not adopted:

* Per-image focal with up to 12 rounds (`perImageFocalRounds: 12`, object-real): the alternation stops at round 8 with
  c_y +0.43 px from the centre (GT −1.0): pp 1.46 px, rotation 0.118° (6 rounds: 0.87 px, 0.077°) at the same
  reprojection error (0.566 px). The 6-round cap stops the slow drift along the orbit's pitch ↔ c_y mode near the truth
  on object-real / abl-object-breath; it is not convergence (ARCHITECTURE "Real-capture robustness", Limits).
* Gyro re-calibration on the rolling-shutter-corrected live model before the second readout pass (tag `rev2`): the
  offset calibrated on the distorted model is biased by where the features sit in the frame (yard-real −30.5,
  abl-yard-rs −27.9 ms; truth −25) and becomes −25.1 / −24.9 ms with the true bias, but accuracy was neutral
  (object-real 2.79e-4 / 0.071°, abl-yard-rs 2.50e-4 / 0.06°, abl-object-rs 2.27e-4 / 0.04°) and worse on yard-real
  (same live model: pp 0.4 → 1.3 px, rotation 0.10 → 0.13°).

#### Principal-point prior in the per-image focal rounds (2026-10, measured, not adopted)

Weak prior on the shared principal point in every shared re-refinement of the per-image focal alternation
(`engineConfig.perImageFocalPPSigmaPx` / `perImageFocalPPAnchor`, BA `principalPointPriors`; off by default). Same command
as above (`--no-dense --no-mesh`, CPU features, worker), tags `ppbase` (current code, no prior) and `pp*`; GT principal
point (+1.50, −1.00) px from the centre on both datasets. "stop" = the 6-round cap or the 0.25 px tolerance; c_y offset
of the shared camera per round.

| dataset, variant | stop | c_y per round (px) | pp px | rot° p50 | ATE / extent | focal % |
|---|---|---|---|---|---|---|
| object-real, no prior (`ppbase`) | cap 6 | −3.52 −2.48 −1.67 −1.03 −0.53 −0.15 | 0.90 | 0.078 | 2.84e-4 | +0.09 |
| object-real, σ 2.5 px at the input pp, cap 6 (`pp25in-r6`) | cap 6 | −3.54 −2.57 −1.82 −1.25 −0.80 −0.48 | 0.59 | 0.055 | 2.81e-4 | +0.11 |
| object-real, σ 2.5 px at the input pp, ≤ 20 rounds (`pp25in-r20`) | tolerance, round 7 | … −0.44 −0.19 | 0.86 | 0.074 | 2.84e-4 | +0.09 |
| object-real, σ 2.5 px at the image centre, ≤ 20 rounds (`pp25c-r20`) | tolerance, round 8 | … −0.06 +0.21 +0.42 | 1.45 | 0.116 | 2.92e-4 | +0.05 |
| object-real, σ 1 px at the input pp, ≤ 20 rounds (`pp10in-r20`) | tolerance, round 5 | −3.70 −2.94 −2.43 −2.09 −1.86 | 0.91 | 0.079 | 2.79e-4 | +0.20 |
| abl-object-breath, no prior (`ppbase`) | cap 6 | −5.58 −4.55 −3.70 −2.98 −2.38 −1.86 | 0.92 | 0.078 | 2.65e-4 | −0.06 |
| abl-object-breath, σ 2.5 px at the input pp, cap 6 (`pp25in-r6`) | cap 6 | −5.62 −4.64 −3.85 −3.20 −2.67 −2.24 | 1.27 | 0.105 | 2.67e-4 | −0.04 |
| abl-object-breath, σ 2.5 px at the input pp, ≤ 20 rounds (`pp25in-r20`) | tolerance, round 9 | … −2.26 −1.92 −1.64 −1.41 | 0.52 | 0.052 | 2.60e-4 | −0.09 |

The prior does not make the alternation converge: the c_y steps still shrink geometrically by almost the same factor per
round (object-real 0.78 → 0.76 at σ 2.5 px, abl-object-breath 0.84 → 0.81), so every run still ends by a stopping rule.
Extrapolated fixed points (last c_y + last step · ρ/(1 − ρ)): object-real +1.2 → +0.6 px, abl-object-breath ≈ +0.9 →
−0.4 px, against −1.0 px GT. With the 6-round cap the prior helps object-real and hurts abl-object-breath (it slows a
drift that is still short of the truth there). With ≤ 20 rounds both improve, but only because the 0.25 px tolerance is
crossed at a better place (rounds 7 / 9) — another early-stopping rule tuned on the same two datasets. Anchored at the
image centre it is worse. σ 1 px contracts faster (factor 0.67; fixed point ≈ −1.4 px, set by the anchor — the input pp,
which the breathing biases by a dataset-dependent amount) but still stops by the tolerance (−1.86 px: pp 0.91 px) and
doubles the focal error. Kept: the 6-round cap without a prior (ARCHITECTURE "Real-capture robustness", Limits).

#### Final verification (2026-10-17, worker engine, CPU features, fast preset)

Sparse: `node e2e/run-e2e.mjs <dataset> --engine worker --no-dense --tag verify --set 'engineConfig={"featureBackend":"cpu"}'`;
dense: the same with `--dense-size 480 --tag verify-dense480` (instead of `--no-dense`), run one after another. Every
`thresholds.json` check passes; the sparse part of the dense runs matches the sparse runs (yard-real ATE 3.18e-4 vs 3.17e-4).

| dataset | kf reg (model) | ATE / extent | rot° p50 | focal % | pp px | scale % (source) | grav° | head° | Δt ms | rolling shutter | dense P / C / F | dense acc p50 / p90 | wall s (refine / densify) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| object | 92/92 (live) | 2.10e-4 | 0.08 | +0.04 | 0.8 | +0.3 (imu) | 0.01 | −0.83 | −24.9 | none | – | – | 133 (59 / –) |
| yard | 80/80 (live) | 2.19e-4 | 0.05 | +0.07 | 0.4 | +0.1 (imu+gps) | 0.01 | −0.64 | −24.9 | none | – | – | 140 (56 / –) |
| object-real | 89/89 (live) | 2.84e-4 | 0.08 | +0.09 | 0.9 | −0.9 (imu) | 0.02 | −0.54 | −25.1 | cols −21.6 ms | 0.9995 / 0.9941 / 0.9968 | 0.31 / 1.06 mm | 943 (86 / 775) |
| yard-real | 81/81 (live) | 3.17e-4 | 0.09 | +0.16 | 0.4 | +1.6 (imu+gps) | 0.06 | −0.93 | −25.1 | rows 22.0 ms | 0.961 / 0.351 / 0.515 | 8.8 / 47 mm | 843 (69 / 684) |
