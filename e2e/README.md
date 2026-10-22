# Synthetic datasets and evaluation

Rendered image sequences with exact ground truth (poses, intrinsics, IMU, GPS, compass,
depth, dense surface samples) for headless end-to-end tests of the whole pipeline
(features → SfM → inertial/GPS alignment → dense), and a scorer for the results.

* Browser code: `src/harness/synth/**` (three.js, WebGL2), page `synth.html`.
* Generator: `node e2e/generate-synth.mjs` (Vite dev server + headless Chromium/SwiftShader).
* Scorer: `node e2e/metrics.mjs` (plain Node, no browser).
* Real captures: raw capture recordings from the phone (Setup › Options › Record raw capture),
  imported with `node e2e/import-recording.mjs`, replayed verbatim and scored without ground truth.
  See "Raw capture recordings".
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

A raw capture recording uses the same layout without the ground truth. It adds `frames.jsonl` (the
verbatim FrameInputs), `live.jsonl` and `recording.json`. See "Raw capture recordings".

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
  ±2 cm sway, looking outward with ±25° slow panning. `trajectory.lateralOffset` (m, + =
  outward) walks the same loop beside the gravel path (off-trajectory test sets).

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
    It uses the focus set when present, otherwise the base set inside the ROI, and every result point
    (accuracy / precision use at most `--max-points`, default 3 M, read with a stride).
  * F-score combines precision and completeness.
  * τ defaults to `meta.denseTau`: 1 % of `sceneExtent` for object (about 1.35 cm), 0.1 m for yard.
* focus box (object isolation; `meta.focus` is the box of the object alone, without the table, its
  placemat or the mug): with `focusPoints` (run-e2e passes the `mesh.ply` vertices), mapped by the camera
  Sim3, `focus.outsideFraction` is the share outside `meta.focus` ± 2 cm and `focus.completeness` the share of
  the visible `gt_focus` samples inside `meta.focus` with a point within τ. With `objectRoi` in the result
  (the engine's object box, `result.mesh.objectRoi`), `objectRoi` reports its IoU with `meta.focus`
  (48³-sample estimate of the Sim3-mapped box), the share of the focus box it covers, whether it contains
  `meta.focus` shrunk by 1 cm, and the plane height under it in the GT frame; `present: false` when the
  engine proposed no box.

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
`minViews` 2, normal 15°, `removeDepthHalos` on the final maps; see ../docs/research/dense-tuning.md "Dense tuning
results" for every experiment): object P 0.9997, C 0.987, F 0.994 (mesh F 0.963 → 0.987); yard
P 0.981, C 0.352, F 0.518 (mesh F 0.348 → 0.492). With the sky test of `removeDepthHalos`
(default since 2026-10, ../docs/research/dense-tuning.md "Sky / low-texture halo removal"; yard maps saved under
`yard-skyb`): yard P 0.9854, C 0.3521, F 0.5188, sky-coloured points farther than τ 5,722 → 707,
mesh F 0.492 → 0.495; object unchanged (P 0.9997, C 0.9874, mesh F 0.9873 → 0.9868).

## Gaussian splats with ground-truth poses (`run-splat-gt.mjs`)

`node e2e/run-splat-gt.mjs` trains the 3D Gaussian Splatting trainer of `src/splat` (src/splat/index.ts,
../docs/research/splat-tuning.md) without the SfM engine: exact poses and intrinsics, so the scores measure the splat
optimisation (and its initialisation), not SfM. Held-out frames are scored as novel views.

```sh
node e2e/run-splat-gt.mjs [<dataset> ...] [--max-size N] [--iterations N] [--max-gaussians N]
     [--init cloud|gt|random] [--cloud PATH] [--init-count N] [--sparse N] [--shell N] [--shell-radius R]
     [--train-stride N] [--train-frames i,j,...] [--test-every N] [--eval-at i,j,...] [--sh 0|1]
     [--no-colour-correction] [--no-mcmc] [--profile] [--panels N] [--tag NAME] [--out DIR] [--set key=json ...]

# MVS initialisation: fused cloud of a dense-GT run (GT frame) at the 'fast' dense size
E2E_GPU=host node e2e/run-dense-gt.mjs object-real --preset fast --max-size 800 --views 40 --tag splat-fast --no-render
E2E_GPU=host node e2e/run-splat-gt.mjs object-real --cloud e2e/out/dense-gt/object-real-splat-fast.ply --sparse 20000 --iterations 3000
E2E_GPU=host node e2e/run-splat-gt.mjs yard-real --cloud e2e/out/dense-gt/yard-real-splat-fast.ply --sparse 30000 \
     --shell 20000 --shell-radius 200 --iterations 3000
```

The page is `splat-gt.html` (`src/harness/splat-gt/main.ts`); `window.runSplatGT(config)`:

1. Training frames: every `--train-stride`-th frame (default 2: 100 of object-real's 200 frames, 120 of yard-real's
   240) or `--train-frames`; test frames: every `--test-every`-th (default 4) of the others (25 / 30 views).
2. Each JPEG is undistorted to pinhole with the GT k1, k2 and the frame's GT focal length (`f` in frames.json of the
   -real datasets; `--set perFrameFocal=false` uses `meta.intrinsics.f`) at `--max-size` (default 800 = the fast
   dense size: 600×800 / 800×600; 960 = full 720×960 / 960×720).
3. Initial Gaussians (`initFromPointCloud`: 3-NN scale, normal-oriented discs, opacity 0.5, SH band 0 from the colour):
   * `--cloud PATH`: a fused point cloud PLY in the GT frame, randomly subsampled to `--init-count` (default half of
     `--max-gaussians`). A dense-GT cloud covers only what MVS reconstructs (object-real: the table region, 6 % of
     the room; yard-real: completeness 0.27 in the ROI). `--sparse N` adds N GT surface samples over the whole scene,
     coloured from the frames with exact GT depth maps (every 20th frame): a stand-in for the engine's sparse SfM
     points, which do reach the room walls and the distant houses.
   * `--init gt`: GT surface samples only (as above); `--init random`: uniform in `meta.roi`, random colours.
   * `--shell N --shell-radius R`: N Gaussians on a sphere of radius R around the cameras' centroid, coloured from the
     first training view that sees them (the sky; `backgroundShell`).
4. `SplatTrainer` (`--max-gaussians`, default 300 000; `--set trainer.*` passes any `SplatTrainerOptions`, e.g.
   `--set trainer.inFlight=2`, `--set 'trainer.lr={"mean":3.2e-4}'`, `--set 'trainer.resolution=[]'`) runs in chunks
   of 100 iterations; at every `--eval-at` iteration (default 0, 500, 1000, 3000, 7000 up to `--iterations`) the test
   views are rendered and scored on their valid pixels: PSNR and SSIM (11-tap Gaussian window, the loss's definition)
   raw and after a per-view per-channel affine colour fit ("colour-fit", removes the unknowable exposure / white
   balance of a held-out -real frame). Training time excludes evaluation.
5. `--profile`: per-pass GPU times from timestamp queries (forces one command buffer in flight), reported per
   evaluation segment.

Output (`e2e/out/splat-gt/`): `<name>.json` (config, frames, every evaluation point with per-view scores, training
speed and log, GPU ms per pass, memory plan, learned colour matrices vs the GT exposure × white balance),
`<name>.splat.ply` (trained model, INRIA layout: opens in SuperSplat / Brush / Spark) and
`<name>/test-NNNNNN.png` (GT | render | |error| of `--panels` test views).

### Splat results (2026-10, Apple M4 GPU, Metal, headless Chrome 155)

Defaults unless noted: SH degree 0, 300 000 Gaussians max, MCMC, per-view colour correction, ¼ → ½ → full
resolution at 10 % / 30 % of the iterations, one command buffer in flight. Initialisation: the dense-GT fused cloud
(`--preset fast --max-size 800 --views 40`, 3.17–3.4 M points, 150 000 used) + sparse GT points (object 20 000, yard
30 000) + for the yard a 20 000-Gaussian sky shell at 200 m. Test views: 25 (object) / 30 (yard) held-out frames.
"PSNR / SSIM" = raw render with the mean training-view colour matrix (≈ identity); "colour-fit" = after a per-view
per-channel affine fit. Training time excludes evaluation; the host was shared with other GPU jobs (load 3–10), so
times vary ±15 %.

| run | size | 0 it | 500 | 1 000 | 3 000 | 7 000 | colour-fit @ 3k / 7k | SSIM @ 7k | time to 3k / 7k |
|---|---|---|---|---|---|---|---|---|---|
| object-real | 600×800 | 19.10 | 26.17 | 28.50 | 30.43 | **30.86** | 32.61 / **33.30** | 0.948 | 48 s / 177 s |
| object-real | 720×960 | 18.94 | 26.08 | 27.88 | 29.72 | 30.06 | 31.53 / 32.02 | 0.930 | 50 s / 160 s |
| yard-real | 800×600 | 17.15 | 25.59 | 27.90 | 29.50 | **29.88** | 30.45 / **30.93** | 0.843 | 23 s / 73 s |
| yard-real | 960×720 | 17.04 | 25.19 | 27.07 | 28.50 | 28.89 | 29.25 / 29.70 | 0.803 | 27 s / 87 s |

For comparison, today's renderings of object-real held-out views (research probe, SfM poses): the lit TSDF mesh
9.2 dB full frame (61 % coverage; 21.9 dB on covered pixels), the unlit mesh 24.8 dB on covered pixels, points + EDL
15.8 dB on covered pixels. Iteration 0 (the initial Gaussians rendered as splats) is already 19.1 dB full frame.

Ablations (3 000 iterations, 600×800 / 800×600; PSNR · colour-fit PSNR · SSIM at 3k; the GT-sample, random, no-shell
and SH 1 rows were scored before the mean colour matrix was applied, which moves PSNR by ≤ 0.15 dB):

| change | object-real | yard-real |
|---|---|---|
| reference (above) | 30.43 · 32.61 · 0.942 | 29.50 · 30.45 · 0.829 |
| init: GT samples only (98 k, no MVS cloud) | 29.82 · 31.78 · 0.934 | |
| init: random in the ROI (150 k) | 20.24 · 20.76 · 0.848 | |
| no sky shell | | 27.98 · 29.52 · 0.819 |
| no colour correction | 31.67 · 32.15 · 0.942 | |
| SH degree 1 | 30.34 · 32.67 · 0.943 | |
| colour correction lr 1e-2, pull 1e-3 (default 2e-3, 1e-2) | 30.49 · 32.67 · 0.943 | |
| 500 000 Gaussians max (263 k at the start) | 30.66 · 32.95 · 0.946 (78 s, 38 it/s) | |
| `inFlight: 2` | same quality; 44 vs 46–51 it/s overall (the pre-encoded next iteration already hides most of the host time) | |

* MVS seeding matters: +0.6 dB over GT samples of the same scene (the MVS cloud is denser on the textured
  near field), +10 dB over random initialisation at this iteration budget.
* Without colour correction the raw held-out PSNR is 1.2 dB higher but the colour-fit PSNR 0.5 dB lower: test frames
  sit between training frames on a trajectory along which the exposure drifts smoothly, so an uncorrected model can
  bake exposure into viewpoint-dependent geometry that happens to suit the neighbouring test frames. The learned
  matrices follow the GT exposure × white balance (correlation of log M_cc with log(gain·wb_c): 0.85–0.90, slope
  0.13–0.22 in sRGB). The correction stays on by default for real captures; an off-trajectory test set would decide.
* SH degree 1 does not help on these Lambertian scenes (it costs 8 % memory).

Speed (fixed 300 000 Gaussians, object-real 600×800, full-resolution iterations, GPU timestamps per pass):
project 0.5 ms, depth sort 1.0, binning 0.6, tile sort 0.7, rasteriser 3.7, loss 1.6, rasteriser backward 10–11,
projection backward + Adam 1.5: **≈ 20 ms GPU per iteration**, 39–43 it/s with one command buffer in flight (the
rest is the completion round trip), 48 it/s with two. Backward history: 256-thread workgroups 12.0 ms → 64-thread
workgroups 10.0 ms (32-thread 13.8, 128-thread 10.9; two splats per thread step 15.7); the float CAS adds cost ≈ 1 ms;
exact tile culling and the σ early-out were within noise on these scenes. Yard iterations are ~2× cheaper (0.29 M vs
0.64 M tile intersections).

## Engine end-to-end runs (`run-e2e.mjs`)

`node e2e/run-e2e.mjs` replays a dataset through the whole engine like the app does, then scores the
result. It starts Vite and headless Chromium (SwiftShader WebGPU), opens `harness.html` once per
dataset and calls `window.runPipeline(config)`. The browser code is in `src/harness/engine/`.

```sh
node e2e/run-e2e.mjs [object|yard|all|<dataset> ...] [--engine worker|inline|mock]
     [--quality fast|balanced|max] [--mode object|scene|auto] [--stride N] [--start N]
     [--max-frames N] [--fps F] [--no-imu] [--no-gps] [--no-heading] [--hfov DEG|gt|none]
     [--no-dense] [--no-mesh] [--dense-size N] [--dense-views N] [--mesh-voxel V]
     [--mesh-triangles N] [--splat] [--splat-iterations N] [--splat-max N] [--splat-size N] [--splat-views N]
     [--splat-previews]
     [--no-export | --export fmt,fmt] [--nvs] [--nvs-views N] [--mapping inline|background]
     [--timeout-min M] [--out DIR] [--tag NAME] [--thresholds FILE] [--no-thresholds]
     [--enforce] [--mock-speed X] [--set key=json ...] [--quiet] [--verbose] [--headed]
     [--device desktop|phone|local-bridge] [--public-url URL] [--port N]

node e2e/run-e2e.mjs all --engine mock                                  # plumbing check, ~40 s
node e2e/run-e2e.mjs object --stride 4 --max-frames 40 --no-dense       # short real run
node e2e/run-e2e.mjs yard --dense-size 480 --dense-views 30 --timeout-min 240
E2E_GPU=host node e2e/run-e2e.mjs object-real --splat --nvs              # + Gaussian splats, scored as novel views
node e2e/run-e2e.mjs object-real --device phone --splat                 # in Safari on an iPhone (see "On a real iPhone")
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
flight. Then it runs `finishCapture`, `refine`, `densify`, `mesh`, with `--splat` the Gaussian-splatting stage
(`engine.splat()`, after densify / mesh, or after refine with `--no-dense`; `--splat-iterations`, `--splat-max`,
`--splat-size`, `--splat-views`, `--splat-previews` set `SplatOptions` and imply `--splat`) and the exports, collecting the
progress events. A failing step is recorded in `errors`, the steps that depend on it are skipped,
and a partial result is still written. On `--timeout-min` the runner calls `harnessAbort()`, which
cancels the long operation, and saves what was done.

**Continued capture (`--extend-at F [--extend-skip-s S]`).** Reproduces "Add more views" after a
page reload (`EngineOptions.extend`, docs/ENGINE.md "Extend"): the first F of the planned frames
are delivered, then `finishCapture` and `refine` end that segment, and the engine is initialised
again with `{ sessionId, resume: true, extend: true }` (forces `persist`: the session is restored
from IndexedDB). S seconds of frames are skipped, and the rest is delivered with every time —
frame `t`, IMU, GPS fix — moved so that the first one is at 500 ms, like a reloaded page's
`performance.now()` (the IMU of that frame is its last second; a GPS fix older than the split is
withheld). `index` keeps counting, so cameras still map to dataset frames. Then `finishCapture`,
`refine` and the rest as usual, on all keyframes. `result.extend` reports the split, the first
refine, the restored keyframes and the second segment's keyframes, how many the live model
registered and how many are in the final model (also logged as `extend: …`).

`EngineClient` timeouts are raised for SwiftShader: init 600 s, frame 300 s, long operations
30 min of inactivity. Set them with `--set timeouts.frameMs=…`.

### Output (`e2e/out/e2e/<dataset>[-tag]/`)

| File | Content |
|---|---|
| `result.json` | Harness result plus the `metrics.mjs` input fields. See below. |
| `sparse.ply`, `dense.ply`, `mesh.ply` | Streamed from the page through `window.__e2eWrite` (Playwright), or the HTTP bridge with `--device phone` / `local-bridge`. |
| `splats.ply` | With `--splat`: `exportModel('ply-splat')` (INRIA 3DGS PLY, run world frame); `--nvs` scores it as the `splat` renderer. |
| `previews/NNN-it<i>.png` | With `--splat-previews` (`SplatOptions.preview`): the training snapshots of the splat stage (the training view nearest the middle photo, ≤ 400 px, raw trainer colours). |
| `splats.spz` | With `--splat` and the `spz` export (in the default export list): `exportModel('spz')` (SPZ v3); `nvs.mjs --splat-file splats.spz` scores the decoded SPZ. |
| `metrics.json` | `computeMetrics` output, sparse-cloud accuracy, live-trajectory metrics and threshold checks (+ the `nvs` summary with `--nvs`). |
| `nvs/nvs.json` | With `--nvs [--nvs-views N]`: the novel-view evaluation of the output (see "Novel-view evaluation"). |

The harness fields of `result.json` are:

* `config` (resolved), `dataset`, `engine` (init result, respawns, status changes; `init.adapter` is
  the engine's adapter string from its own realm).
* `device` (`src/harness/engine/device.ts`): `userAgent`, `platform`, `hardwareConcurrency`, `devicePixelRatio`,
  `screen`, `secureContext`, and the page's WebGPU `adapter` (vendor, architecture, description, features,
  `limits.maxBufferSize` (the RAM class on Apple GPUs) and other limits; requested only, no device), or `adapterError`.
* `runner`: `wallSeconds`, `error`; device runs add `device`, `waitSeconds` (until the page connected) and
  `page` (its user agent and URL).
* `timings`: harness steps in ms. `stageTimings`: per engine progress stage, with the
  `src/app/stages.ts` UI stage and its duration.
* `frames`: one record per delivered frame, with the decision, guidance, tracking, counters,
  overlay count, engine `timings`, wall and decode ms, and any error.
* `decisions`, `guidance` and `tracking`: counts.
* `live`: keyframes, registered images, points, `timeToInit`, frame and keyframe ms stats, the
  last live cameras, and their `coverage`: the capture screen's object-mode ring
  (`src/app/coverage.ts`) — `stable`, `completeness`, covered sectors per elevation band
  (`covered`: low −10–10°, mid 10–40°, high > 40°, of 12 each), the `prompt` and the per-cell
  keyframe `counts`. Elevation bands need gravity (`useImu` + `useGravity`), else `elevation: false`
  (azimuth only). A scene capture usually stays `stable: false`.
* `progress`: decimated progress events.
* `sparse`: intrinsics normalised to the dataset size, mapped `images`, `metric`, `stats`,
  `pointsCount`, and the `unmapped` / `duplicates` camera ids.
* `dense`, `mesh` and `exports` (sizes only), plus `logs`, `errors` and page `memory`.
* `extend` (with `--extend-at`): `firstFrames`, `resumeIndex`, `skippedFrames`, `rebaseMs`, `first`
  (keyframes, live-registered, refined images, refine ms, error), the re-init `init`,
  `resumedKeyframes`, `secondKeyframes`, `secondRegisteredLive`, `secondRegisteredFinal`.
* `dense`, `mesh` and `exports` (sizes only), plus `logs`, `errors` and page `memory`. `mesh` also
  carries `objectRoi` (the engine's object box, null without one) and the mesh stage's `stats` (TSDF
  `voxelSize`, `tsdfAutoVoxelSize`, `tsdfCoarsenLevels`, `tsdfBlocks` / `tsdfMaxBlocks`, `objectRoi` as text
  or `none: <reason>` / `off`, `objectRoiConfidence`, `meshingMs`). `--no-object-roi` meshes the whole scene
  in object mode (the baseline of the focus-box scores).
* `splat` (with `--splat`): the `SplatResult` (count, iterations, ms, `stats`: views, size, init, it/s, memory
  plan, checkpoints, …), `lastDetail` (the last training progress `detail`) and the `splats.ply` path; also
  copied to `metrics.json`. With `--splat-previews`: `previews` (iteration, arrival ms after the stage
  started, PSNR against that training photo, file) and `firstTrainingMs`; the summary line gives the first
  arrival, the largest gap while training and the PSNR first → last, next to `stats.splatPreviewFinalPsnr`
  (the trained model rendered from the same view in the same raw colours: what the previews converge to).

The `metrics.mjs` fields are `intrinsics`, `images`, `metric`, `numInputImages`, `pointsFile` and (with a
mesh) `objectRoi`.
To re-score a run, use `node e2e/metrics.mjs e2e/data/object e2e/out/e2e/object/result.json`.

**SparseCamera → dataset frame.** Each camera is mapped by its `frameIndex` (the `FrameInput.index`
it came from, an extension field of the engine's `SparseCameraEx`), then by its `t` (nearest
delivered frame within 1 ms), then by its `id` read as the keyframe ordinal. The ordinal is the
k-th frame answered with `'keyframe'`. `sparse.images[].via` records which rule was used.
Engine intrinsics at another resolution are rescaled with the §2 rule. Transposed sizes are
flagged in `sparse.intrinsicsNote`.

**Diagnostics** (off by default; large results):

* `--set liveTrace=true` (any engine): `result.liveTrace` keeps the camera poses of every
  `LiveUpdate` that carries the model (frame index, keyframe / registered / point counts, and the
  coverage ring of that update without the counts). Poses are the registered keyframes sorted by
  id, so they map to `live.keyframeIndices` only while every keyframe is registered.
* `--engine inline --set debugDump=true --set 'engineConfig={"mapping":"inline",…}'`:
  `result.debug` (`src/harness/engine/debug-dump.ts`, reads engine internals) has, after every
  keyframe frame, each keyframe's live registration state, 3D-point count and pose
  (`keyframeTrace`), and after finishCapture and after refine the live model plus every verified
  pair (keyframe ids, frame indices, model, inliers, median triangulation angle, R, t). Comparing
  R / t with the GT relative pose separates wrong pairs from wrong registrations.
* `--set 'densifyRepeat={"overlap":false}'` (any `DenseOptions`): after densify, the same sparse
  model is densified again with these options and the clouds are compared bit for bit
  (`result.denseRepeat`, log `densify repeat … bit-identical to the first` or the first
  difference; stage timings under step `densifyRepeat`). Two separate runs cannot be compared
  that way: the sparse model already differs between runs of the same code (see "Dense CPU/GPU
  overlap" below).

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
skipped with `--no-dense`. The object-isolation checks `meshFocusOutside`, `objectRoiIoU` and
`objectRoiContainsFocus` (1 = the box contains `meta.focus` − 1 cm) apply to mesh runs with the object box on
(skipped with `--no-mesh` or `--no-object-roi`), `meshFocusCompleteness` to every mesh run. object-real has the
object box acceptance as limits: IoU ≥ 0.6, the box contains `meta.focus` − 1 cm, mesh vertices outside
`meta.focus` ± 2 cm ≤ 5 %. `meshFocusCompleteness` has no limit: compare it with a `--no-object-roi` baseline run
(within 0.01), and compare `voxelSize` / `meshingMs` rather than `tsdfCoarsenLevels` (the bounded TSDF takes its
auto voxel from the depths inside the box, so the coarsening ladder starts elsewhere).

The exit code is 0 when every pipeline finished without errors and every enforced check passed,
1 otherwise, and 2 for usage errors. Mock runs report the checks but do not enforce them unless
`--enforce` is given; `--no-thresholds` stops enforcing them for any engine.

`metrics.json` also holds two informational blocks:

* `sparseCloud`: accuracy of `sparse.ply`, with the same definition as the dense accuracy.
* `live`: ATE and rotation error of the last live keyframe poses. They are mapped by keyframe
  order, which assumes that `LiveUpdate.cameras` lists every keyframe in capture order.

### By hand

Open `harness.html?dataset=object&engine=mock&maxFrames=40&densify=1` on the dev server
(`npm run dev`). It runs on load and shows the result. The query keys (`src/harness/engine/query.ts`) are
`engine`, `quality`, `mode`, `stride`, `maxFrames`, `start`, `fps`, `imu`, `gps`, `heading`, `headingNorth`,
`densify`, `mesh`, `splat`, `splatIterations` (implies `splat`), `exports`, `mockSpeed`, `engineConfig=<JSON object>`
(e.g. `{"featureBackend":"cpu"}`) and `config=<JSON>`, a whole `HarnessConfig` that the single keys override.
Malformed input stops the page with an error instead of running a default: bad JSON, an unknown `engine`,
`quality`, `mode` or `headingNorth`, a number key that is not a number, `splatIterations` below 1. From the console,
`await runPipeline({...})` returns the same `HarnessResult`. When `window.__e2eWrite` is missing, the dense
cloud is embedded as base64 in `densePoints`.

### On a real iPhone (`--device phone`)

The same harness page runs in a Safari tab on the phone, on the phone's GPU (Metal), memory limits and jetsam.
Nothing on the phone is automated: you open one URL and keep the tab in front. The results come back to the
computer through an HTTP bridge on the dev server and are scored exactly like a desktop run. Needs Safari on
iOS 26+ (WebGPU), the phone on the computer's network (or a tunnel), and the dataset in `e2e/data/` on the
computer (the phone fetches it from the dev server).

```sh
# once: a certificate the phone trusts (WebGPU, motion sensors and storage need a secure context)
brew install mkcert && mkcert -install
mkdir -p e2e/out/cert && mkcert -cert-file e2e/out/cert/dev.pem -key-file e2e/out/cert/dev-key.pem \
  "$(ipconfig getifaddr en0)" "$(scutil --get LocalHostName).local" localhost 127.0.0.1
#   AirDrop "$(mkcert -CAROOT)/rootCA.pem" to the phone, install the profile (Settings > General > VPN & Device
#   Management) and enable full trust (Settings > General > About > Certificate Trust Settings).

export E2E_HTTPS_CERT=e2e/out/cert/dev.pem E2E_HTTPS_KEY=e2e/out/cert/dev-key.pem
node e2e/run-e2e.mjs object-real --device phone --splat               # → e2e/out/e2e/object-real-phone/
node e2e/run-e2e.mjs object-real --device phone --engine mock          # plumbing check (also works over plain HTTP)
# or behind an HTTPS tunnel to the port (no certificate on the phone; Vite then listens on 127.0.0.1 only)
node e2e/run-e2e.mjs yard-real --device phone --public-url https://<name>.trycloudflare.com --port 5180
```

* **What happens.** The runner starts Vite on `0.0.0.0:5180` (`--port`) with the bridge plugin
  (`e2e/lib/e2e-bridge.mjs`; added to the runner's inline config only, `apply: 'serve'`, never in `vite.config.ts`
  or a build) and prints `https://<LAN address>:5180/__e2e/go` for each dataset. That link redirects to
  `harness.html?bridge=1&run=<id>&config=<the whole HarnessConfig>`. The page has no Playwright binding, so it
  installs the fetch bridge (`src/harness/engine/bridge.ts`, the run's flow in `bridge-run.ts`): progress lines,
  page errors and console warnings go to the runner's stdout; `sparse.ply`, `dense.ply`, `mesh.ply` and `splats.ply` are POSTed as raw bytes
  (16 MB requests, no base64) into the run's directory; at the end the page posts `{ result, error }`. Then the
  runner scores it with the desktop code (`scoreRun`: metrics, `thresholds.json` checks, `--nvs`, summary) and
  moves on to the next dataset. The page shows a "Next run" link to `/__e2e/go`.
* **Output.** The same layout as a desktop run, in `<dataset>-phone/` (the tag defaults to the device name;
  `--tag` overrides it). `result.json` has the phone's `device` block (user agent, adapter, `maxBufferSize`,
  `wakeLock`: whether the screen lock was held when the result was sent),
  the engine's `init.adapter`, `stageTimings`, and for `--splat` `splat.stats.splatItPerSec` (ms per iteration
  = 1000 / it/s) and `splatMemPeakMB`.
* **Keep the phone awake.** Set Settings > Display & Brightness > Auto-Lock > Never for the session (and back
  afterwards): a locked phone suspends the Safari tab, its timers, fetches and GPU work. The page also requests a
  screen wake lock once the runner accepts it. Whether WebKit wants a tap for that is [U]; if the request fails,
  the runner prints a warning and the page shows a "Keep the screen on" button that asks again. The page polls
  the runner every 5 s. The runner warns after 60 s without contact (screen locked, Safari in the background,
  or the tab was killed) and says when the page is back.
* **Timeouts.** At `--timeout-min` the runner asks the page to abort and scores the partial result. If the
  pipeline has not returned 90 s after the page saw the abort (an engine call that ignores it), the page sends
  `harnessSnapshot()` with the error `timeout (snapshot of the unfinished run)`, as the desktop runner does. Only a
  page that sends nothing within 120 s of the abort ends the run without a result.
* **Reloads and second tabs.** Each tab keeps a random token in `sessionStorage`, and the runner remembers the
  token of the tab that started the run. When that tab comes back during the run, it was reloaded (iOS reloads
  a tab that jetsam killed, "A problem repeatedly occurred"), so the run fails with that message instead of
  running again. Any other tab is refused and the run goes on, even when it is reloaded. This covers a second
  tap on the link or a link opened in a new tab. Whether `sessionStorage` survives a WebContent kill is [U]; if
  not, the reloaded tab is refused like a second tab and the run ends at the timeout. Once the result is in,
  `/__e2e/go` shows "No e2e run is waiting" until the runner registers the next dataset (it may still be
  scoring); reload it then.
* **Security.** While the runner runs, the dev server is open to the whole LAN, just like `npm run dev`. The
  run's random id (crypto, 128 bit) only stops other web origins: a page on another site cannot read it from
  `/__e2e/go`'s redirect, so it cannot write files or post a result. It does not stop another host on the
  network, which can `curl -i …/__e2e/go` and read the id from the `Location` header. The bridge writes only
  under that run's output directory (`safeJoin`: no absolute paths, no `..`), at most 512 MB per request and
  4 GiB on disk per run. It forgets the run once its result is in. Vite's Host check (plain HTTP only; Vite skips
  it over HTTPS) allows IP addresses, localhost, this machine's host name and, with `--public-url`, the
  tunnel's host name. Run it only on a network you trust.
* **Proving the path without a phone.** `--device local-bridge` runs the identical server and page code with
  headless Chromium here opening the `/__e2e/go` URL (no bindings; HTTPS too when `E2E_HTTPS_*` is set). On a
  12-frame object set (`generate-synth.mjs object --frames 12`; 2026-10-22, SwiftShader) it matched the desktop
  runner exactly: the mock engine gave byte-identical `sparse.ply` / `dense.ply` / `mesh.ply`, the worker engine
  (`--no-dense --set 'engineConfig={"featureBackend":"cpu"}'`) a byte-identical `sparse.ply`, both identical
  metrics and checks. The nightly CI runs `run-e2e.mjs all --engine mock --device local-bridge`. A renderer
  crash or a closed page fails the run at once (as on the desktop), not at the timeout.
* **Limits.** The phone loads the dev server's unbundled modules, so it measures the engine on real Metal and
  real jetsam but not the production worker bundle (the cd7a534 class needs a built-harness run). Runs are
  manual: note battery level, charger and thermal state with each result, and do not quote a warm phone's
  numbers as sustained. Measured numbers go into a dated `docs/research/` report and a measured row under
  "iPhone estimate (not measured)" below; none has been recorded yet.

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

### Dense on the real GPU (2026-10, M4 Metal, worker engine)

```sh
E2E_GPU=host node e2e/run-e2e.mjs object-real --engine worker [--quality balanced] --tag dense-after-fast \
     --set 'engineConfig={"denseGpuTimestamps":true}'
```

Every frame, IMU + compass (+ GPS for yard), GPU features, dense at the preset size (fast: 40 views at 800 px;
balanced: 80 views at 960 px = the datasets' size, fusion step 2). "before" = the code before the dense work
(+ the stage timing log and GPU timestamps), "after" = footprint-adaptive point cap, multi-scale passes,
GPU-resident PatchMatch data flow (upload caches, GPU finalisation / prior, 12 B/px readback), native float16,
serpentine passes, image LRU up to every view (../docs/research/dense-tuning.md "Dense on a real GPU"). Each before / after pair ran
back to back (host load 4–8, other agents' GPU jobs running: GPU-pass wall time is 1.2–1.6× the timestamped GPU
time; exception: the object-real fast "before" ran at load 18–23 while a unit-test run overlapped, so that pair's
times favour "after"). densify = depth maps + fusion; GPU = PatchMatch execution time from timestamp queries.

| run | densify s | depth maps s (view loads / PatchMatch / halo / store) | GPU s | fusion s (fuse / cap) | points | dense P / C / F | acc p50 / p90 | checks |
|---|---|---|---|---|---|---|---|---|
| object-real fast, before | 60.7 | 47.3 (12.7 / 17.2 / 9.7 / 6.4) | 7.8 | 12.3 (10.7 / 1.2) | 3.29 M → 0.57 M | 0.9981 / 0.9966 / 0.9973 | 0.44 / 1.8 mm | pass |
| object-real fast, after | 51.1 | 37.7 (7.5 / 14.8 / 9.9 / 4.1) | 6.6 | 12.2 (9.6 / 2.4) | 3.34 M → 1.76 M | 0.9995 / 0.9968 / 0.9982 | 0.29 / 0.99 mm | pass |
| yard-real fast, before | 47.4 | 38.1 (10.4 / 14.5 / 6.0 / 5.7) | 5.5 | 8.3 (7.4 / 0.7) | 4.07 M → 0.75 M | 0.803 / 0.408 / 0.541 | 25 / 175 mm | **FAIL** acc p50 |
| yard-real fast, after | 36.4 | 26.8 (6.3 / 10.6 / 5.6 / 3.2) | 4.7 | 8.8 (6.6 / 1.9) | 4.12 M → 1.83 M | 0.935 / 0.397 / 0.557 | 8.8 / 62 mm | pass |
| object-real balanced, before | 181.0 | 154.9 (35.7 / 76.7 / 23.0 / 15.8) | 47.8 | 24.8 (21.6 / 2.7) | 4.82 M → 1.43 M | 0.9987 / 0.9989 / 0.9988 | 0.37 / 1.4 mm | pass |
| object-real balanced, after | 148.3 | 122.2 (22.6 / 61.6 / 23.6 / 10.2) | 39.8 | 24.7 (20.2 / 3.9) | 4.86 M → 3.54 M | 0.9995 / 0.9989 / 0.9992 | 0.28 / 0.87 mm | pass |
| yard-real balanced, before | 148.6 | 127.8 (29.5 / 64.2 / 14.9 / 15.7) | 36.2 | 19.7 (17.3 / 1.8) | 6.30 M → 2.27 M | 0.898 / 0.493 / 0.636 | 14 / 102 mm | pass |
| yard-real balanced, after | 125.9 | 105.0 (23.9 / 52.1 / 15.2 / 10.0) | 31.0 | 19.7 (16.1 / 3.0) | 6.33 M → 3.55 M | 0.953 / 0.483 / 0.641 | 7.3 / 49 mm | pass |

The after-balanced clouds (> 3 M points) are scored with the completeness fix below (the harness's own table printed
C 0.9986 / 0.433 from every 2nd point). Cap lifted (`--set 'engineConfig={"dense":{"maxPoints":30000000}}'`, after):
object-real fast (3.29 M points) 0.28 / 0.86 mm, P 0.9998, C 0.9969; yard-real fast (4.07 M) P 0.945, C 0.405, F 0.567,
p50 8.3 mm; yard-real balanced (6.3 M) P 0.962, C 0.494, F 0.653, p50 7.1 mm (completeness on every point).

* The yard-real `denseAccuracyMedian` check (≤ 20 mm) failed at `fast` only because of the global-voxel point cap
  (25 mm); it passes with the footprint LOD (8.8 mm; 8.0 mm in a second run).
* Kernel diet (`--set 'engineConfig={"dense":{"patchMatch":{"patchRadius":6,"patchStep":3,"affineWindow":true,"numSamplesPerIter":3}}}'`,
  fast, with multi-scale): GPU object-real 7.2 → 3.9 s, yard-real 6.3 → 3.1 s; yard-real P / C / F 0.942 / 0.401 /
  0.563 → 0.932 / 0.396 / 0.555, p50 8.0 → 9.8 mm; object-real p50 / p90 0.29 / 0.98 → 0.32 / 1.1 mm, C 0.9969 →
  0.9981. Balanced yard-real: GPU 31.0 → 15.4 s, depth maps 105.0 → 87.3 s, densify 125.9 → 111.2 s, P / C / F
  0.953 / 0.483 / 0.641 → 0.946 / 0.483 / 0.639, p50 7.3 → 7.8 mm. Not in the presets (a quality cost at fast,
  within ~1 pt of precision at balanced: a candidate for balanced / max after an on-device check).
* `metrics-core` scores completeness on every ROI point now; accuracy and precision still use a ≤ `maxPoints`
  (3 M) strided subsample. Clouds above 3 M points only appear with the new point cap at balanced / max (the global
  voxel kept ≤ 2.3 M).

### Dense CPU/GPU overlap (2026-10-22, SwiftShader, worker engine)

```sh
node e2e/run-e2e.mjs object-real --dense-size 480 --no-mesh --no-export --tag dov-<x> \
     --set 'engineConfig={"featureBackend":"cpu","mapping":"inline"}' [--set 'densifyRepeat={"overlap":false}']
```

Base `18d73c9` ("before") vs the overlap ("after"), back to back, then one run with the same-model A/B
(`densifyRepeat`). 4 shared cores (load 2.5–13); SwiftShader runs PatchMatch on the CPU, so it is 99 % of the
stage and its time follows the host load. Engine thread = what the engine worker itself spends on each step.

| run | depth maps s | engine thread: view loads / pass inputs / PatchMatch / halo / store s | helpers: decodes / halo + serialise s | fusion s | points | dense P / C / F | checks |
|---|---|---|---|---|---|---|---|
| before | 961.7 | 2.7 / 1.0 / 952.5 / 3.9 / 1.5 | — | 3.7 | 1 269 875 | 0.9995 / 0.9923 / 0.9959 | pass |
| after | 819.2 | 0.2 / 1.6 / 816.5 / 0.0 / 0.1 | 5.2 / 3.3 | 4.2 | 1 269 989 | 0.9994 / 0.9921 / 0.9958 | pass |
| A/B, overlap on | 1143.6 | 0.2 / 2.6 / 1139.6 / 0.0 / 0.2 | 5.9 / 5.2 | 5.3 | 1 269 253 | 0.9995 / 0.9928 / 0.9961 | pass |
| A/B, overlap off (same model) | 1037.4 | 2.8 / 1.2 / 1027.1 / 4.6 / 1.7 | — | 8.4 | 1 269 253, bit-identical | (same cloud) | |

Before / after differ in the sparse model (the captures diverge at the live initialisation, frame 10: 700 vs
750 points, refine 12 079 vs 12 094 points), hence the 114-point difference; on one model the clouds are
bit-identical. The wall-time differences are SwiftShader PatchMatch time (the same-model 'on' run took 10 %
longer); the pass inputs (IndexedDB reads) rose 1.2 → 2.6 s with the overlap, the reads contending with the
writes in flight. The overlap's effect on a real GPU is unmeasured: the M4 table above bounds it at ≈ −52 to
−55 % depth maps at fast if the helpers keep up, and an `E2E_GPU=host` `densifyRepeat` A/B at fast and balanced
is still to run.
Analysis: ../docs/research/dense-tuning.md "CPU/GPU overlap of the depth-map stage".

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

What changed (details and the evidence for each choice: ../docs/research/real-capture-robustness.md):
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
orbit (../docs/research/real-capture-robustness.md "Rolling shutter"); the translational term removes it.

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
dense accuracy p50 ≤ 6e-4 of the extent (0.8 mm) / 2 cm, completeness ≥ 0.98 / 0.28; object-real also the
object box checks above.

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
  on object-real / abl-object-breath; it is not convergence (../docs/research/real-capture-robustness.md "Per-image focal (focus breathing)").
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
doubles the focal error. Kept: the 6-round cap without a prior (../docs/research/real-capture-robustness.md "Per-image focal (focus breathing)").

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

#### Phone-like analysis rate (`--stride 3`, 2026-10-22, worker engine, CPU features)

`node e2e/run-e2e.mjs <dataset> --stride 3 [--start 1|2] --no-dense --set 'engineConfig={"featureBackend":"cpu"}'`
analyses 2 (yard) / 2.7 (object) frames per second, the rate a phone may reach.

- Before = claude/next-level 7598a6e.
- After = the second-review state:
  - re-triangulation of the registered neighbours of a keyframe at the edge of the map;
  - gyro-aided registration only in refine's offline SfM (`gyroAidedRegistration: 'offline'`),
    after P3P failed on every candidate, and refused when no verified pair checks the baseline;
  - the offline SfM for live models with an unclosed loop.
- Runs were back to back. Every run, the diagnosis and the stride-1 checks:
  ../docs/research/real-capture-robustness.md "Phone-rate capture", "Second review".
- The "gyro-aided" column is refine's `stats.gyroAidedRegistrations` in result.json (also
  self-metrics `gyro.aidedRegistrations`); "off" = path not configured.

| dataset | live kf reg | lost frames | offline SfM | gyro-aided | ATE / extent | rot° p50 | failed checks |
|---|---|---|---|---|---|---|---|
| yard-real, before (3 runs) | 19/48 | 38 of 80 | run (29 not registered) | – | 5.10–5.12e-4 (offline) | 0.108–0.111 | – |
| yard-real, after (4) | 48/48 | 0 | skipped | off live | 5.30–5.32e-4 (live) | 0.119 | – |
| yard-real `--start 1`, before (4) | 20/49 | 38 | run (3 models) | – | 6.61e-4 (offline, 33 of 49) | 0.219–0.224 | registrationRate, gravityErrorDeg |
| yard-real `--start 1`, after (4) | 48/48 | 0 | skipped | off live | 9.53–9.63e-4 (48 of 48; 7.0–7.2e-4 on the 32 common frames) | 0.135–0.139 | – |
| yard-real `--start 1`, after, offline SfM forced (2) | 48/48 | 0 | run (1 model of 48; 3 with the path off) | 1 offline | 9.63e-4 (live kept) | 0.139 | – |
| yard-real `--start 2`, before / after (2 + 2) | 20/51 / 34/51 | 39 / 17 | run | – / 0 offline | 1.78e-3 / 1.84e-3 (offline, 37 of 51) | 0.349 / 0.403 | registrationRate, ateRelative, rotErrDegMedian |
| object-real, before / after (1 + 1) | 50/50 | 0 | skipped | – / off live | 6.00e-4 / 6.07e-4 | 0.192 / 0.188 | – |

Accuracy is not accepted on these runs. ATE / extent and rotation p50 are worse than before at
stride 3 in every yard phase:

- from frame 0: ATE +4 %, rotation +7–10 %;
- from frame 2: ATE +3 %, rotation +15 %;
- from frame 1, on the 32 comparable frames: ATE +6–8 %, while the rotation p50 halves.

`--start 1` sits at the 1e-3 `ateRelative` limit: 2 of 14 runs of this and the previous review
state went over it (1.03e-3, 1.035e-3), and none of the 8 here did. The change delivers registration
at a phone's rate (no lost frames), not accuracy. `--start 2` loses the loop's corner both before
and after: it has no map points to register on. The stride-3 `scaleError` failure of 2026-10-21 is
gone on both sides (the keyframe-interval IMU scale of claude/next-level).

### Refine performance on Metal (2026-10)

Real GPU (`E2E_GPU=host`: headless Chrome 155 on an Apple M4, Metal), worker engine, fast preset, GPU features. The
host was shared with other heavy jobs (load 6–19 during these runs, ±30–50 % timing noise), so the baseline and the
final code ran back to back. Baseline = `19448e5` (the code before this work plus its timing stats); final = this
branch (../docs/research/refine-performance.md). Per-phase times come from the new `stats` fields (`matchingMs`,
`vocabularyMs`, `verifyMs`, `matchWaitMs`, `liveRefineMs`, `rsMs` / `rsSearchMs` / `rsReadjustMs` / `rsEvaluations`,
`offlineSfMMs`, `principalPointMs`, `perImageFocalMs`; `offlineSfM` = run / skipped and why).

```sh
E2E_GPU=host node e2e/run-e2e.mjs object-real --engine worker --tag refine-final-dense       # dense + mesh
E2E_GPU=host node e2e/run-e2e.mjs yard-real --engine worker --no-dense --tag refine-final2   # refine iterations
```

Wall time (s); reconstruction = live refinement / rolling shutter / offline SfM / principal point / per-image focal:

| run (load) | refine | matching | reconstruction | total |
|---|---|---|---|---|
| object-real dense, baseline (9–11) | 232.8 | 30.3 | 201.8 = 3.3 / 80.4 / 63.7 / 3.7 / 50.5 | 402 |
| object-real dense, final (10–11) | **44.8** | 12.4 | 31.9 = 1.1 / 12.1 / – / 2.0 / 16.5 | 194 |
| yard-real dense, baseline (8–11) | 101.1 | 35.3 | 65.4 = 3.4 / 12.6 / 42.9 / 1.3 / 4.9 | 200 |
| yard-real dense, final (8) | **29.9** | 15.4 | 14.2 = 2.9 / 6.2 / – / 1.0 / 3.9 | 135 |
| object-real sparse, baseline ×2 (5–7 / 10–12) | 153.0 / 229.6 | 27.4 / 28.8 | 125.2 / 200.5 | 191 / 278 |
| object-real sparse, final ×2 (9–10 / 13–19) | 53.9 / 149.9 | 13.5 / 26.0 | 40.1 / 122.7 | 113 / 211 |
| yard-real sparse, baseline ×2 (7 / 10) | 96.7 / 115.0 | 32.1 / 43.9 | 64.2 / 70.6 | 135 / 175 |
| yard-real sparse, final ×2 (9 / 16–19, offline SfM run) | 28.5 / 177.7 | 14.7 / 42.5 | 13.4 / 134.6 | 66 / 285 |

Matching (final, object-real): vocabulary 3–4 s (was 9–10: quantisation on 3 CPU workers), verification on the workers
(14–16 s of worker time; the engine thread waits ≈ 1 s for results), GPU matcher 6–9 s (was idle during every CPU
verification). Rolling shutter: 17–18 subset BAs in ≈ 10 parallel rounds (was 28 in sequence). The second final yard-real
sparse run is the slow path on purpose: under load 16–19 the live model left the loop open (50 px gap), the health check
said so (`unclosed loop? 4322 correspondences on 69 loop-like and 5644 on 128 split pairs (limits 420 / 2517)`), the
offline SfM ran and won (87.4 vs 83.1 % explained) — what the baseline does in that case.

Accuracy (same runs; every `thresholds.json` check passes as before — the yard-real dense accuracy p50 ≤ 2 cm check fails
for both, 25 / 24 mm, as it already did in the pre-work Metal run):

| run | reg | ATE / extent | rot° p50 | focal % | pp px | scale % | grav° | head° | Δt ms | rolling shutter | model | sparse p50 | dense P / C / F | dense p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| object-real dense, baseline | 90/90 | 2.86e-4 | 0.051 | +0.139 | 0.47 | −0.45 | 0.021 | −0.45 | −25.3 | cols −21.3 ms | live | 0.60 mm | 0.9980 / 0.9962 / 0.9971 | 0.47 mm |
| object-real dense, final | 90/90 | 2.82e-4 | 0.053 | +0.130 | 0.49 | −0.37 | 0.021 | −0.45 | −25.3 | cols −21.4 ms | live | 0.60 mm | 0.9981 / 0.9963 / 0.9972 | 0.44 mm |
| object-real sparse, baseline ×2 | 90/90 | 2.89 / 2.87e-4 | 0.050 / 0.051 | +0.159 / +0.160 | 0.46 / 0.45 | −0.41 / −0.42 | 0.021 | −0.45 | −25.3 / −25.2 | cols −21.3 / −21.4 | live | 0.60 mm | – | – |
| object-real sparse, final ×2 | 90/90 | 2.88 / 2.84e-4 | 0.054 / 0.053 | +0.150 / +0.148 | 0.51 / 0.49 | −0.40 / −0.45 | 0.021 | −0.45 | −25.2 / −25.3 | cols −21.4 / −21.3 | live | 0.59 / 0.60 mm | – | – |
| yard-real dense, baseline | 81/81 | 3.11e-4 | 0.074 | +0.157 | 0.68 | +0.98 | 0.061 | −0.91 | −25.1 | rows 22.0 ms | live | 20.5 mm | 0.803 / 0.408 / 0.541 | 25.1 mm |
| yard-real dense, final | 81/81 | 3.12e-4 | 0.075 | +0.161 | 0.65 | +0.97 | 0.061 | −0.91 | −25.1 | rows 21.9 ms | live | 20.4 mm | 0.807 / 0.409 / 0.543 | 24.5 mm |
| yard-real sparse, baseline ×2 | 81/81 | 3.15 / 3.07e-4 | 0.074 / 0.076 | +0.163 / +0.165 | 0.62 / 0.71 | +0.97 / +0.82 | 0.061 | −0.91 | −25.1 | rows 21.9 / 22.1 | live | 20.9 / 20.6 mm | – | – |
| yard-real sparse, final ×2 | 81/81 | 3.02 / 2.49e-4 | 0.074 / 0.080 | +0.163 / +0.160 | 0.63 / 0.48 | +0.94 / +1.06 | 0.061 / 0.062 | −0.91 | −25.1 | rows 21.9 / 22.1 | live / offline | 20.4 / 19.1 mm | – | – |

Unclosed live loops on purpose (`--set 'engineConfig={"livePairs":{"retrieval":0,"spatial":0}}'`, yard-real, sparse;
tags `refine-noloop` (baseline), `refine-h2-noloop`, `refine-final-noloop`, `refine-final-noloop2`; load 9–24): whether
the live model still closes the loop through refine's pairs varies by run. Final code: two runs closed (116 / 958 and
98 / 1755 loop-like / split correspondences, 86.8 / 86.4 % explained) → offline SfM skipped, live model, ATE 2.98 / 3.46e-4,
rotation 0.075 / 0.053°; one unclosed (98 / 2956: a gap beyond 0.4 f, caught by the split total; 85.1 % explained) → offline
SfM run and chosen (87.6 %), ATE 2.31e-4 — as the baseline run (offline chosen, 85.3 vs 87.9 %, ATE 2.44e-4). Every
unclosed live model of these runs (gaps 12–15 px, 50–255 px and beyond 0.4 f) exceeded at least one limit (by 1.2–10×);
every closed live or offline model stayed ≥ 1.3× below both.

## Raw capture recordings (real iPhone data, `import-recording.mjs`)

A recording is a real capture saved on the phone so a desktop can replay it through the identical
engine: every `FrameInput` the app sent to the engine, with the pixels as a JPEG. It is a dataset
without ground truth, so it is scored GT-free.

**Recording (phone).** Setup › Options › **Record raw capture** (developer option, off by default).
From the engine's init until the capture ends, `CaptureRecorder` (`src/app/recorder.ts`) is a
`FrameTap` on the engine client (`src/app/frame-tap.ts`). It sees every frame the engine client
accepts, not the ones the FramePump skipped while the engine was busy. For each frame it sends the
pixels and every `FrameInput` field to the recorder worker (`src/workers/recorder.worker.ts`). The
worker encodes a JPEG at q 0.9, at most 1280 px on the long side. It writes to OPFS through sync
access handles, or to IndexedDB where OPFS is missing (`src/io/recording-store.ts`). A frame the
engine client rejected before posting it (engine disposed or stopped after repeated failures,
worker not startable) is flagged `notSent` in `live.jsonl`: the engine never got it.

* **Size:** about 0.1–0.35 MB per frame. At the pump's ≤ 10 Hz that is 1–3 MB/s.
* **Sensor log:** next to the frames, `sensors.jsonl` records the raw DeviceMotion and
  DeviceOrientation events with every field iOS gives (rotation rate, both accelerations,
  `interval`, `webkitCompassHeading` / `webkitCompassAccuracy`, arrival time and event
  `timeStamp`) and every GPS fix. It is posted to the worker in 500 ms batches, an estimated
  10–20 KB/s. It is for characterising the sensors (rates, jitter, duplicates, GNSS accuracy:
  docs/research/ios-sensors.md §8), and it also holds the IMU drained into frames the capture
  dropped while paused. Replays do not use it.
* **Free space:** a recording needs 1 GB of free origin quota to start (the capture screen warns at
  300 MB). It stops by itself on low storage, a write error, or a worker that falls 8 s behind
  (time with the page hidden does not count: iOS suspends the worker too). This stops only the
  recording, not the scan, and a toast says so.
* **Back-pressure:** when more than 3 frames are in the recorder worker, the frame just sent
  resolves its `processFrame` only when the worker caught up, so at most 4 are in it. The
  FramePump then skips camera frames as when the engine is busy, and the recording still equals
  the engine's input. `live.jsonl` keeps the wait per frame.

**Export (phone).** Diagnostics › **Raw capture recordings** lists each recording with its frames,
duration and size. **Export** builds STORE ZIP parts of at most 200 MB each, because the iOS share
sheet reads every file fully into memory (about 2×, docs/research/ios-platform.md §3). For each part,
tap **Prepare**, then **Share / Save**. The GPS fixes are left out unless **Include location** is
on. **Delete** frees the space.

**Import (desktop).**

```sh
node e2e/import-recording.mjs ~/Downloads/sfm-rec-20261021-140210-ab12cd34-part*.zip   # → e2e/data/sfm-rec-20261021-140210-ab12cd34/
node e2e/import-recording.mjs <parts or a folder of them> --name kitchen-1 [--out DIR] [--force]
```

The importer extracts every part into one directory. It checks the CRCs and the entry names, that
no part is missing, and that every frame in `frames.jsonl` has its JPEG (rows recorded without
pixels, `image: null`, are only counted). The dataset name must be a plain name that does not
start with '.', so `--force` can only replace a directory inside the output directory. Extracting
the parts with any unzip tool into one folder gives the same result.

**Layout** (the synthetic layout without ground truth, plus the verbatim stream):

| File | Content |
|---|---|
| `meta.json` | `scene: 'recording'`, `source: 'sfm-raw-capture'`, `gt: null`, `intrinsics: null`, `sceneExtent: null`; the recorded `mode`, `quality`, `hfovPriorDeg`, `createdAt` (the replay defaults); `width × height` (first recorded JPEG), `srcWidth × srcHeight` (what the engine got), `fps` (delivered rate), `numFrames` (frames a replay sends), `duration`, `screenAngle`, `originLatLonAlt` (first fix, null without location), `locationStripped`, `camera` (no device ids), `app`, `recorder`, `summary`, `missingImages` (frames recorded without pixels), `notSent`, `badLines` |
| `frames.jsonl` | one line per frame sent, in order: `{seq, sentAt, input, image: {file, width, height, srcWidth, srcHeight, kind, bytes}}`. `input` is the `FrameInput` without pixels, VERBATIM: `t, screenAngle, screenAngleSince, imu, geo, heading, index, segment, gravityDevice, attitude, imuCalibrationVersion, timeSource, rawBuffer, maxSide`. It is JSON-safe (`src/io/serialize.ts`): typed arrays are `{"$typed": "Float64Array", "data": […]}`, and −∞ / NaN / −0 are `{"$num": "-Infinity"}`. `image: null` (with `error`): the engine got the frame, but its pixels could not be recorded. `notSent: true` (from `live.jsonl`): the engine never got the frame. |
| `frames/NNNNNN.jpg` | the pixels the engine got, ≤ 1280 px, in the same pixel layout. A raw sensor-landscape VideoFrame stays landscape, and `rawBuffer` says so. `kind` is VideoFrame / ImageBitmap / RGBAImage. |
| `frames.json` | `[{index, file, t}]` of the frames a replay sends (the engine got them, with pixels): `index` is `FrameInput.index`, or `seq` when those are not unique. There is no `pose`. |
| `imu.json`, `geo.json`, `heading.json` | derived from the frames the engine got (those without pixels included): the IMU slices concatenated, the distinct GPS fixes, and the per-frame headings with `north` / `declinationDeg`. This is what the engine got; the raw sensor streams are `sensors.jsonl`. |
| `live.jsonl` | the device's answer per frame: `{seq, index, ms, waitMs, update: {decision, guidance, tracking, numKeyframes, numRegistered, numPoints, timings, debug}, error, notSent?}` |
| `sensors.jsonl` | the raw sensor log, one reading per line: `{k: 'motion', t, ts, interval, rot: [α, β, γ], acc: [x, y, z], accG: [x, y, z]}`, `{k: 'orientation', t, ts, absolute, alpha, beta, gamma, absoluteFlag, compassHeading, compassAccuracy}`, `{k: 'geo', t, fix}`. `t` is the arrival time (performance.now()), `ts` the event's `timeStamp`, null a value the platform did not give. The `geo` rows are left out unless the export includes the location. |
| `recording.json` | the recorder header: app / device, EngineOptions, camera, capture settings, stop reason, and the summary (frames, bytes, sensor rows, FramePump stats) |

**Replay.** `node e2e/run-e2e.mjs <name>` works like any dataset, but in VERBATIM mode
(`src/harness/engine/pipeline.ts`, `feed.ts verbatimInput`):

* Each `FrameInput` is the recorded one, in the order it was sent. There is no synthesized screen
  angle, segment, time source or gravity, and no WMM heading conversion. The bitmap is the decoded
  JPEG.
* The recorded mode, quality, hfov prior and date are the defaults. `--mode`, `--quality` and
  `--hfov` override them.
* `--start`, `--stride` and `--fps` still select frames. The IMU samples of the skipped rows are
  prepended to the next replayed frame, as the FramePump drains everything since its last frame;
  with `--start` the first frame gets those of every row before it.
* A frame recorded without pixels cannot be sent. Its IMU samples go to the next replayed frame,
  because the device engine had them. A `notSent` frame is left out with its IMU, because the
  device engine never got it.
* `--no-imu`, `--no-gps` and `--no-heading` withhold a stream the way the app would without that
  sensor: no samples and calibration version −1; no fix; no heading and no attitude. Without GPS
  the compass also goes back to magnetic north: a heading the app had turned to true north with
  the declination of a fix becomes `heading − declinationDeg`, flagged `north: 'magnetic'`, and an
  absolute attitude is rotated back by the same declination.
* `--set replaySize='"original"'` resizes each JPEG back to the size the engine got: a VideoFrame
  fitted to its `maxSide`, else the bitmap size.
* `--set verbatim=false` synthesizes the inputs from `frames.json` / `imu.json` like a synthetic
  dataset.
* `result.recording` counts the rows replayed (`rows`), the rows not replayed (`skipped`: no pixels,
  not sent, unparseable) and the `notSent` ones. With `live.jsonl` it also holds the device's
  decision histogram, its last counts, its frame ms and recorder wait, and the per-frame agreement
  of the replay's decisions with the device's (`rate`, keyframes on each side, and a
  `device>replay` confusion count).

**GT-free scoring.** A dataset whose `meta.source` is `sfm-raw-capture`, or whose frames lack
`pose`, gets self-consistency metrics instead of `computeMetrics` (`e2e/lib/self-metrics.mjs`;
`e2e/metrics.mjs` does the same when re-scoring). They are:

* registered / input images and the registered fraction;
* mean reprojection error and mean / median track length;
* gyro breaks and outliers of the model refine chose, and deregistered gyro outliers;
* explained correspondences, and whether offline SfM ran;
* the scale source and uncertainty, gravity / north alignment with the yaw source, and the
  camera–IMU time offset;
* the focal against the hfov prior;
* live init, lost frames and frame ms;
* the device comparison.

Every threshold check is skipped ('no ground truth'), and so is `--nvs`. The exit code reflects
pipeline errors only, and the summary row shows `SKIP`. These are proxies: a gyro-consistent
model can still be off in scale or shape, so accuracy claims need ground truth.

**Fidelity limits.**

* The JPEG is a re-encode of what the engine converted. A 'balanced' capture on iOS gets native
  1920 × 1440 frames but is recorded at 1280 × 960, so features come from a smaller image unless
  you use `replaySize: 'original'` (which upsamples).
* Background keyframe mapping depends on timing. Compare decisions with `--mapping inline` and
  expect approximate agreement.
* Recording costs time on the device: the main-thread clone / blit and the worker encode. The
  per-frame `waitMs` and the FramePump stats in `recording.json` show it.

**UI check.** `node e2e/ui/recorder.mjs` runs Setup (record on, VideoFrame grab) → capture with the
mock engine and the fake camera → Diagnostics → Export (Prepare + Save) → `import-recording.mjs`
→ Delete.

**Plumbing run (2026-10-22, headless Chromium, this machine).** The UI flow recorded 11 frames
(VideoFrame grab, 1920 × 1080 recorded at 1280 × 720, 160 KB, one 0.16 MB part). Every frame
waited 0 ms for the recorder except the first (19 ms in total). The FramePump skipped 5 frames as
busy.

* `run-e2e.mjs ui-recording --engine mock` replayed the 11 frames verbatim. Decisions agreed with
  the device's for 11 of 11 (7 keyframes on each side). The GT-free table printed, all 8 checks
  were skipped, and the run exited 0 in 5 s.
* `--engine worker --no-dense` ran the real engine on SwiftShader on the same 11 verbatim frames,
  at 4.6 s per frame (p50). The fake camera shows a flat test pattern, so the engine made no
  keyframe: all 11 frames were 'skip', and they agreed with the mock device's decisions for 4 of
  11. Refine then failed ("needs at least 2 keyframes"), as it would on the phone. The GT-free
  scoring still wrote metrics.json without a crash, and the exit code was 1 for the pipeline
  error.

**Not yet measured on an iPhone:** the encode cost per frame next to the engine, the OPFS write
rate, how many frames back-pressure costs at 10 Hz, the sensor log's event rates and size, and
sharing 200 MB parts from a Safari tab and from a Home Screen app.

## Novel-view evaluation (`nvs.mjs`)

`node e2e/nvs.mjs` scores what the user sees. It renders a finished run (the output directory of
`run-e2e.mjs`) at held-out dataset frames, the frames that did not become keyframes, and compares
the images with the ground-truth photos. This is the yardstick for photoreal stages such as
3D Gaussian Splatting.

```sh
node e2e/nvs.mjs <dataset> <runDir> [--renderers points,points-edl,mesh-lit,mesh-unlit,splat]
     [--views N] [--frames i,j,...] [--test-dataset NAME|DIR] [--max-size N|dense|full] [--pose relative|sim3]
     [--focal relative|keyframe|shared] [--rs auto|run|gt|none] [--rs-rotation-only] [--keep-blurred]
     [--background R,G,B] [--no-ssim] [--common] [--save-images] [--out DIR] [--verbose] [--headed]

E2E_GPU=host node e2e/nvs.mjs object-real e2e/out/e2e/object-real --save-images
E2E_GPU=host node e2e/run-e2e.mjs object-real --nvs [--nvs-views N]   # pipeline, then the evaluation
```

`<dataset>` is a name under `e2e/data` or a dataset directory. The run directory needs
`result.json` plus `dense.ply`, `mesh.ply` and/or `splats.ply`. The page is `nvs.html`, the browser
code is in `src/harness/nvs/`, and the pure parts are `e2e/lib/nvs-views.mjs` and
`e2e/lib/nvs-metrics.mjs` (unit-tested in `src/harness/nvs/nvs-lib.test.ts`). With `run-e2e.mjs
--nvs`, the summary also goes into `metrics.json` as `nvs`. It is informational and never fails a
run.

### Test views

* **Held-out frames**: every frame that has no image in the result and is not in
  `live.keyframeIndices`. Frames the capture rejected as blurred are left out (`--keep-blurred`
  keeps them). `--views` (default 24) of them are spread evenly over the sequence. `--frames`
  picks frames by index; keyframes are allowed, which gives a training-view score. On the
  baselines below there are 101 (object-real) and 154 (yard-real) candidates, each 1–3 frames
  from a keyframe.
* **Pose** (`--pose relative`, the default): take the nearest registered keyframe k (by frame
  index) with its estimated pose. Compose it with the GT relative motion from k to the frame, and
  divide the translation by the Sim3 scale (Umeyama run → GT on the camera centres). Local SfM
  error around k carries over, as it would for k itself. `--pose sim3` maps the GT pose through
  the global Sim3 instead.
* **Off-trajectory views** (`--test-dataset NAME`): the frames of another dataset of the same
  seeded scene, e.g. the same orbit 12 cm higher, replace the held-out frames. Generate one with
  `generate-synth.mjs <preset> --name <name> --set trajectory.<field>=…` (object: `radius`,
  `heightAbove`; yard: `height`, `lateralOffset`, which walks beside the loop without moving the
  gravel path; `center` / `semiAxes` move the path and are refused). The tool refuses a test
  dataset with another scene, seed, texture scale, camera or yard loop. Its GT poses share the
  world frame, so `--pose sim3` (the default here) maps them through the run's Sim3; `--pose
  relative` and the focal use the registered keyframe nearest in space. Every frame is a candidate,
  and ω, ν and the GT image come from the test dataset. Output goes to `<runDir>/nvs-<name>/`.
  Every view (held-out too) records the nearest training camera (GT centre distance and optical-axis
  angle); `nvs.json` `distance` summarises it.
* **Intrinsics**: the run's f, cx, cy, k1 and k2. The focal (`--focal relative`, the default) is
  the nearest keyframe's refined per-image focal (`sparse.images[].f`, normalised to the shared f
  by the median) times f_frame / f_k from the GT. The GT ratio covers the focus breathing of the
  -real sets. `keyframe` drops the GT ratio, and `shared` uses the shared f.
* **Ground-truth image**: the dataset JPEG goes through exactly what the engine does to its dense
  views (`denseViewImage`). It is undistorted with those intrinsics to `--max-size` by
  `undistortImage`. The default, `dense`, is the run's dense view size, taken from the "dense
  memory plan" log line or else from the preset: 800 px (600×800) for fast. Then
  `rsCorrectPinholeImage` corrects the rolling shutter to the frame time, which is the time of the
  GT pose. `--rs auto` / `run` uses the run's readout, and `gt` uses the dataset's true readout.
  Both use ω = R_cam_device · gyro(tTrue), interpolated from `imu.json`. With a GT focus distance
  (the -real sets) they also apply the translational term ν: the camera-frame velocity of the
  centre (central difference of the neighbouring GT frames) divided by the focus distance, as the
  engine does with `rsTranslation`. `--rs-rotation-only` drops ν. Pixels that map outside the
  source are left out of every metric.

### Renderers

Each renderer outputs RGBA8 at the undistorted pinhole camera, with premultiplied alpha and
transparent pixels where nothing is drawn. Two settings make the projection exact: the vertical
fov comes from f, and a three.js view offset places the principal point. The GPU test
`src/harness/nvs/renderers.gpu.test.ts` measures the error against the pinhole projection. The
mean centroid error is below 0.04 px for mesh squares and below 0.1 px for radius-sized splats.
Individual squares are within 0.2 px; the spread comes from MSAA.

| renderer | what is drawn |
|---|---|
| `points` | the app's `Viewer` drawing `dense.ply` as the viewer screen does: round splats sized by the per-point radius, EDL off |
| `points-edl` | the same with Eye-Dome Lighting, the app default when there is no mesh |
| `mesh-lit` | the `Viewer` drawing `mesh.ply` with vertex colours under the hemisphere light and headlight (the app default until the unlit switch below) |
| `mesh-unlit` | the same unlit (vertex colours only), the app default since 2026-10 (Display sheet "Mesh lighting" off) |
| `splat` | `splats.ply` (INRIA 3DGS PLY layout: `x y z nx ny nz f_dc_0..2 f_rest_* opacity scale_0..2 rot_0..3`, run world frame), drawn by `src/splat/nvs.ts`; `--splat-file splats.spz`: the SPZ export, decoded (`decodeSplatSPZ`) and handed over as that PLY, output `<runDir>/nvs-spz/` |

The splat module plugs in through
`export async function createNvsRenderer({ ply, width, height }): Promise<{ render(cam), dispose?() }>`.
`render(cam)` gets `cam = { width, height, fx, fy, cx, cy, R (row-major world-to-camera), t }` and
returns RGBA8 rows top to bottom with premultiplied alpha. The hook in `src/harness/nvs/renderers.ts`
uses `import.meta.glob`, so while the module does not exist the renderer is skipped with a reason.
By default, every renderer whose input file exists in the run directory is used.

### Metrics

Each metric is computed per view, then averaged over the views. The code is in `e2e/lib/nvs-metrics.mjs`.

* **full**: the render composited over black (`--background`) against the GT on every valid
  pixel. Holes count as background.
* **covered**: only pixels with alpha ≥ 0.5, using the un-premultiplied colour.
* **coverage**: covered valid pixels divided by valid pixels.
* **cc** (exposure-compensated): a least-squares 3×4 affine colour transform render → GT is fitted
  per view on the covered pixels and applied to them. Uncovered pixels stay background. This
  removes the global part of the exposure and white-balance drift of the -real sets.
* **PSNR**: 10·log10(255²/MSE), with the MSE over the RGB channels in 8-bit sRGB.
* **SSIM**: 11×11 Gaussian window (σ 1.5), K1 0.01, K2 0.03, computed per RGB channel and then
  averaged (the 3DGS convention). The windows are restricted to the scored mask (normalised
  convolution), and the SSIM map is averaged over the mask.
* **common** (`--common`): every renderer scored on the same pixels, the valid pixels that ALL
  renderers of the run cover (`scoreCommon`: PSNR / SSIM of the un-premultiplied colour, and the
  PSNR after a per-renderer affine colour fit on them). `covered` uses each renderer's own pixels,
  so a mesh that draws only the well-reconstructed surfaces is scored on those; `common` compares
  the renderers where all of them draw something.
* **shift**: the residual translation render → GT. It is found by mean-removed SSD on 2×
  downsampled luma over ±6 px, with a parabolic sub-pixel step. It is a sanity check of pose,
  intrinsics and projection.

Output goes to `<out>/nvs.json` (default `<runDir>/nvs/`):

* the options, metric definitions, run info (Sim3 scale, rolling-shutter model) and setup
  (point and triangle counts, the WebGL renderer string);
* a per-renderer summary (mean, median, min and max);
* per view: the frame, nearest keyframe and gap, the GT `rsPx` / `blurPx` / gain / WB, the camera,
  render and score ms, and the scores including the affine matrix and the shift;
* the host load.

`--save-images` writes `view-<frame>-<renderer>.png` panels: GT | render | |render − GT| × 3.

Cost: 6–7 s per view for 4 renderers at 600×800 (host load 9). Rendering takes 10–20 ms per image.
Scoring takes about 1 s per renderer in JS and is dominated by SSIM; `--no-ssim` skips it. 24 views
take about 2.5 min.

### Baselines (2026-10, Metal)

These are the current app outputs on the -real sets. The runs are `object-real-metal-fast` and
`yard-real-metal-fast`: `E2E_GPU=host node e2e/run-e2e.mjs <dataset> --engine worker`, fast preset,
dense views 600×800 / 800×600, 40 views. The command is
`E2E_GPU=host node e2e/nvs.mjs <dataset> <runDir> --save-images`, with 24 held-out views at the dense
size, rolling-shutter correction from the run, relative pose and relative focal. The host was an
Apple M4 under load 7–8, and each run took 154–167 s. All values are means over the views. The
shift is the median residual render → GT translation in px.

| dataset | renderer | coverage | full PSNR / SSIM | covered PSNR / SSIM | cc full PSNR | cc covered PSNR / SSIM | shift px |
|---|---|---|---|---|---|---|---|
| object-real (90 keyframes, 101 held-out) | `points` | 0.751 | 10.13 / 0.428 | 23.26 / 0.745 | 10.14 | 23.66 / 0.749 | 0.27 |
| | `points-edl` (app default without a mesh) | 0.751 | 9.35 / 0.147 | 15.13 / 0.218 | 9.84 | 18.41 / 0.355 | 0.32 |
| | `mesh-lit` (app default) | 0.609 | 8.64 / 0.527 | 23.56 / 0.876 | 8.67 | 25.52 / 0.874 | 0.13 |
| | `mesh-unlit` | 0.609 | 8.70 / 0.550 | **28.70 / 0.915** | 8.71 | 30.08 / 0.917 | 0.09 |
| yard-real (81 keyframes, 154 held-out) | `points` | 0.770 | 10.73 / 0.415 | 27.03 / 0.679 | 10.73 | 27.37 / 0.683 | 0.53 |
| | `points-edl` | 0.770 | 10.22 / 0.191 | 18.11 / 0.313 | 10.58 | 22.22 / 0.503 | 0.43 |
| | `mesh-lit` | 0.639 | 9.73 / 0.368 | 23.10 / 0.625 | 9.80 | 25.71 / 0.638 | 0.28 |
| | `mesh-unlit` | 0.639 | 9.83 / 0.382 | **26.90 / 0.657** | 9.83 | 27.26 / 0.658 | 0.35 |

* **Holes dominate.** The mesh covers 61–64 % of the frame and the points 75–77 %. Full-frame PSNR
  stays at 8.6–10.7 dB whatever the colour does: the missing room walls and floor (object, where
  the TSDF is cut at 2× the median depth), and the sky, distant houses and holes (yard). Exposure
  compensation changes it by less than 0.5 dB. Filling the frame is where a splat stage can win
  the most.
* **One-line fixes, measured here** (the unlit mesh is now the default, and the GLB export is unlit too: ../docs/research/viewer-rendering.md).
  * Unlit instead of lit mesh: covered +5.1 dB / +0.039 SSIM on object-real, +3.8 dB / +0.032 on
    yard-real. Full frame +0.06 / +0.10 dB. The lighting is not affine, so the colour fit recovers
    only part of it (cc covered 25.5 vs 30.1 dB).
  * Points without EDL: covered +8.1 / +8.9 dB, full frame +0.8 / +0.5 dB, and full-frame SSIM
    0.147 → 0.428 / 0.191 → 0.415.
  * Neither fix changes coverage.
* **Points vs mesh.** Points cover 13–14 points more of the frame than the mesh. Their covered
  PSNR is 5.4 dB below the unlit mesh on object-real, because of disc edges, gaps and background
  bleeding through, and equal to it on yard-real.

Sensitivity of the harness (same runs; `points` and `mesh-unlit`, PSNR only with `--no-ssim`; one
option changed at a time):

| dataset | variant | mesh-unlit covered PSNR | shift px | points covered PSNR | shift px |
|---|---|---|---|---|---|
| object-real | default | 28.70 | 0.09 | 23.26 | 0.27 |
| | `--rs-rotation-only` | 27.85 | 0.11 | 23.10 | 0.34 |
| | `--rs none` | 27.97 | 0.14 | 23.01 | 0.26 |
| | `--focal keyframe` / `shared` | 28.70 / 28.59 | 0.09 | 23.27 / 23.22 | 0.27 |
| | `--pose sim3` | 28.05 | 0.43 | 23.25 | 0.30 |
| | `--max-size full` (720×960) | 27.90 | 0.10 | 22.87 | 0.36 |
| | all 110 held-out frames incl. blurred | 28.66 | 0.10 | 23.21 | 0.31 |
| | 24 keyframes (training views, `--frames`) | 28.70 | 0.07 | 23.25 | 0.28 |
| yard-real | default | 26.90 | 0.35 | 27.03 | 0.53 |
| | `--rs-rotation-only` | 26.86 | 0.40 | 26.91 | 0.53 |
| | `--rs none` | 26.29 | 0.63 | 25.96 | 0.72 |
| | `--focal keyframe` / `shared` | 26.89 / 26.83 | 0.36 | 27.02 / 26.89 | 0.54 |
| | `--pose sim3` | 26.69 | 0.94 | 26.47 | 0.83 |
| | `--max-size full` (960×720) | 26.21 | 0.40 | 26.31 | 0.62 |
| | 24 keyframes (training views) | 27.23 | 0.30 | 27.37 | 0.46 |

* **The GT image needs the full rolling-shutter correction.** On the object orbit, rotation and
  translation cancel on the object, so a rotation-only correction is no better than none (mesh
  27.85 vs 27.97 dB, points 23.10 vs 23.01 dB), 0.85 dB below the full correction. The yard is a walk with panning, where rotation dominates and the
  correction is worth 0.6–1.1 dB.
* **The keyframe-relative pose beats the global Sim3**: up to 0.65 dB, with a residual shift of
  0.1 vs 0.4 px (object) and 0.35 vs 0.9 px (yard).
* **24 views are representative** (all 110 object-real held-out frames: −0.04 dB).
* **Held-out frames are nearly training views.** They sit 1–3 frames (≤ 0.4 s) from a keyframe:
  the score difference is 0.0 dB on object-real and 0.3 dB on yard-real. This tests interpolation
  between nearby views, not extrapolation to new viewpoints.
* **Full resolution costs 0.4–0.8 dB.** The reconstruction carries the detail of the dense
  resolution, not of the photo.

Limits of the reference: the -real images keep artefacts that no reconstruction reproduces. These
are 18 % corner vignetting, sensor noise, JPEG q 0.85, motion blur (`blurPx` per frame in
`nvs.json`), the per-frame exposure and white-balance drift (the `cc` metrics remove its global
part), and the residual rolling shutter of the first-order correction. So 30 dB is not a hard
ceiling, but the reachable PSNR is well below that of clean renders. The yard keeps a 0.3–0.5 px
residual shift even on training views, so it comes from the reconstruction or the reference, not
from the held-out pose mapping.

### Splat stage in the engine (2026-10, Metal)

`E2E_GPU=host node e2e/run-e2e.mjs object-real --engine worker --splat --nvs --tag int-object` (fast preset: 3 000
iterations, ≤ 200 000 Gaussians, every registered keyframe as a training view at the 600×800 dense size; M4, host load
4–6 from other apps; 207 s for the whole run, every enforced check passes). The splat stage took 60.3 s: prepare
12.6 s (initial Gaussians: 90 113 from the fused cloud's footprint LOD + 12 316 sparse points in 1.1 s; 90 view
decodes / uploads 11.5 s), training 47.5 s (64.6 it/s over the ¼ → ½ → full schedule, 53.8 it/s at full resolution;
6 checkpoints of 22 MB, 168 ms each), finishing 0.1 s; final 200 000 Gaussians, `splats.ply` 13.0 MB. Memory plan
(engine log `splat memory plan`): peak ≈ 811 MB of the 1.2 GB budget, GPU 301 MB. Novel views, same run, 24 held-out
frames:

| renderer | coverage | full PSNR / SSIM | covered PSNR / SSIM | cc full PSNR / SSIM | cc covered PSNR | shift px |
|---|---|---|---|---|---|---|
| `points` | 0.763 | 10.39 / 0.475 | 24.02 / 0.755 | 10.41 / 0.478 | 24.52 | 0.19 |
| `points-edl` | 0.763 | 9.61 / 0.239 | 15.51 / 0.332 | 10.10 / 0.308 | 18.71 | 0.22 |
| `mesh-lit` (app default) | 0.610 | 8.65 / 0.530 | 23.55 / 0.877 | 8.68 / 0.529 | 25.50 | 0.13 |
| `mesh-unlit` | 0.610 | 8.71 / 0.552 | 28.65 / 0.915 | 8.72 / 0.553 | 30.00 | 0.08 |
| **`splat`** | **0.999** | **31.24 / 0.958** | **30.62 / 0.950** | **33.74 / 0.959** | **33.44** | 0.08 |

* The points / mesh rows reproduce the baselines above within 0.3 dB.
* Splats fill the frame (coverage 0.999 vs 0.61–0.76) — +22.6 dB full-frame PSNR over the lit mesh — and also beat the
  unlit mesh on covered pixels by 2.0 dB. Per view the splat full PSNR ranges 23.4–35.1 dB (median 31.1).
* Caveat (see "Held-out frames are nearly training views"): the test frames sit 1–3 frames from a training keyframe,
  which favours an image-fitted representation like splats even more than the mesh; an off-trajectory test set would
  measure extrapolation.
* A 300-iteration smoke run of the same pipeline (`--splat-iterations 300 --no-mesh`, 6 views) already reaches 25.0 dB
  full frame.

### Viewer splat layer (2026-10, Metal)

`e2e/ui/splat-shots.mjs` opens an e2e run directory (`run-e2e.mjs … --splat`: `result.json`, `sparse.ply`,
`dense.ply`, `mesh.ply`, `splats.ply`) in the app's viewer at iPhone size (393×852 @3x, the UI harness): the page
decodes the files, `MockEngine.injectSession` serves them as the scan's models, and the layer pill is tapped through
Splats / Points / Mesh at the same viewpoints — 'open' (the app's default view of splats: from the middle photo,
pulled back 15 %), 'orbit' (that view orbited 40° about the vertical: off the photo path) and 'overview' (the whole
content from above, the viewer's fit). Full-page PNGs go to `--out` (default `/tmp/sfm-viewer-shots/`), plus
`<name>-stats.json` with the splat layer's frame time while orbiting and its GPU memory. Needs a hardware adapter
(the viewer refuses SwiftShader for splats):

```sh
E2E_GPU=host node e2e/run-e2e.mjs yard-real --engine worker --splat --tag viewer-yard     # 170 s, every check passes
E2E_GPU=host node e2e/ui/splat-shots.mjs e2e/out/e2e/yard-real-viewer-yard --name yard-real
E2E_GPU=host node e2e/ui/splat-shots.mjs e2e/out/e2e/object-real-int-object --name object-real
```

| run (fast preset, 200 k Gaussians) | splat canvas | frame median / p90 (submit → done) | intersections | layer GPU |
|---|---|---|---|---|
| object-real `int-object` (90 views 600×800) | 786×1704 | 16.3 / 22.5 ms | 0.93 M | 107 MB |
| yard-real `viewer-yard` (81 views 800×600, sky shell 8 886 Gaussians at r = 160 m; splat stage 42.6 s: prepare 12.9 s, training 29.5 s at 105 it/s) | 786×1704 | 12.7 / 18.1 ms | 0.34 M | 107 MB |

M4, headless Chrome, host load 5–8. Seen side by side, the splats at 'open' and 'orbit' are photoreal and complete
(walls, floor, sky) where the mesh has holes and the points show their splat texture; the overview from far outside
the capture shows floaters and the sky shell as blurred blobs — why the app starts from the photos. The splats and
the unlit mesh line up pixel for pixel at the same pose (the GPU test measures ≤ 0.06 px between Gaussian centroids
and three.js's projection of their means).

### Fly-through video (`e2e/ui/flythrough.mjs`)

The same run directory and page set-up as splat-shots.mjs, then the fly-through MP4 export twice. First through its
pieces in the page (src/app/video-export.ts: the camera path of the sparse model, the splat layer's offscreen
frames, WebCodecs H.264, src/io/mp4.ts). That MP4 is decoded again with VideoDecoder: the frame count, the PSNR of
every 10th decoded frame against the same pose read back directly (encode loss only), wall time per frame, peak JS
heap and the layer's GPU memory. Then through the UI (Export → "Fly-through video (MP4)" → Prepare → Save, a
download). Both files and `<name>-flythrough.json` go to `--out` (default `/tmp/sfm-flythrough/`), counted by
ffprobe when it is installed. It fails outside 180–300 frames or below 34 dB, and warns above 100 ms per frame (the
30 s / 300 frames target). It needs a hardware adapter and Chrome's H.264 encoder (Playwright's Chromium has none).
`--probe` runs only the Diagnostics "Test video encoding" probe, in any browser.

```sh
E2E_GPU=host node e2e/ui/flythrough.mjs e2e/out/e2e/object-real-int-object --name object-real
E2E_GPU=host node e2e/ui/flythrough.mjs e2e/out/e2e/yard-real-viewer-yard --name yard-real
node e2e/ui/flythrough.mjs --probe       # SwiftShader Chromium: frame path 'canvas', no H.264 encoder
```

No host results yet.

## Breakthrough results (2026-10-20, Apple M4, Metal)

**Final state** (`claude/breakthrough` after the review fixes and the depth-supervision default; `--tag final2`,
one run per dataset, load 3.7–12 at start, every enforced check passes). The tables below are the earlier
before / after comparison; the final runs agree with them within run-to-run noise:

| fast preset, M4 | object-real | yard-real |
|---|---|---|
| pipeline total (s): before `eb131af` → final | 236.1–244.9 → 195.3 | 180.0–193.9 → 146.0 |
| capture / refine / densify / mesh / splat (s) | 34.0 / 39.5 / 42.1 / 12.5 / 61.3 | 31.0 / 22.0 / 36.4 / 6.8 / 44.8 |
| splat stage | 200 k Gaussians, 3000 it at 64.6 it/s, prepare 13.7 s, plan 863 MB (GPU 353 MB) | 200 k, 3000 it at 100.2 it/s, prepare 13.7 s, plan 810 MB (GPU 336 MB) |
| ATE / extent, rotation p50, scale error | 2.88e-4, 0.049°, −0.4 % | 3.06e-4, 0.072°, +1.0 % |
| dense P / C / F, accuracy p50 | 1.000 / 0.997 / 0.998, 0.29 mm | 0.933 / 0.396 / 0.557, 8.9 mm |
| held-out full PSNR / SSIM: splat vs `mesh-lit` (old default) | **31.29 / 0.959** vs 8.65 / 0.530 | **31.00 / 0.871** vs 9.81 / 0.378 |
| held-out covered PSNR: splat / `mesh-unlit` / `points` | 30.80 / 28.64 / 24.01 | 30.89 / 26.73 / 28.06 |
| off-trajectory full PSNR, splat (`mesh-lit`) | −5 cm 30.49 (8.97), +5 cm 28.95 (8.39), +12 cm 27.04 (9.55), +25 cm 22.23 (10.73) | in 25.55 (8.97), out 28.07 (10.22), +0.75 m 21.29 (8.91), +1.5 m 18.80 (8.37) |
| off-trajectory `--common` PSNR, splat vs `mesh-unlit` | 29.70 / 30.06 / 29.50 / 28.03 vs 27.53 / 28.39 / 28.49 / 28.34 | 30.45 / 29.00 / 29.01 / 27.08 vs 26.43 / 26.26 / 26.40 / 25.13 |

```sh
E2E_GPU=host node e2e/run-e2e.mjs object-real --engine worker --splat --nvs --tag final2      # and yard-real
E2E_GPU=host node e2e/nvs.mjs object-real e2e/out/e2e/object-real-final2 --test-dataset object-real-up12 \
  --common --renderers mesh-lit,mesh-unlit,splat                                            # each test set
```

Final before / after measurement of the `claude/breakthrough` work (engine splat stage, faster refine,
footprint-adaptive dense). **Before** is the user's main at `eb131af` plus only the `E2E_GPU=host` launch patch
(`bf55e6c`, cherry-picked; no pipeline change). **After** is `claude/breakthrough` at `7e6169c` with the measurement
tooling of this section (`nvs.mjs --test-dataset / --common`, yard `lateralOffset`). Both ran `E2E_GPU=host node
e2e/run-e2e.mjs <dataset> --engine worker` (fast preset; after adds `--splat --nvs`) on the same `object-real` /
`yard-real` files, on an Apple M4 (10 cores, 16 GB) with headless Chrome 155. Jobs ran one at a time, but other apps
and a parallel agent shared the host, so each run's 1-minute load average is given. Every fast-preset pipeline run was
repeated (run 1 / run 2). All novel-view scores come from this branch's `nvs.mjs` at the 600×800 dense size with
`--common`, for the baseline runs too.

### Pipeline

| object-real, fast preset | before (`eb131af`) | after (`claude/breakthrough`) |
|---|---|---|
| host load (1-min mean, run 1 / run 2) | 4.8 / 4.2 | 4.8 / 5.0 |
| pipeline total (s, run 1 / run 2) | 244.9 / 236.1 | 198.5 / 193.2 (without splat 138.8 / 133.7) |
| · capture (live tracking of every frame) | 35.1 / 34.2 | 34.7 / 34.0 |
| · post-capture processing | 206.8 / 199.0 | 158.3 / 153.9 (without splat 98.5 / 94.4) |
| &nbsp;&nbsp;refine (run 1: matching + reconstruction) | 137.9 / 132.8 (25.8 + 111.9) | 40.1 / 38.6 (9.8 + 30.1) |
| &nbsp;&nbsp;densify (run 1: depth maps + fusion) | 55.3 / 52.1 (46.5 + 8.9) | 44.0 / 42.3 (33.4 + 10.6) |
| &nbsp;&nbsp;mesh | 12.6 / 13.1 | 13.2 / 12.5 |
| &nbsp;&nbsp;splat (prepare + training) | — | 59.7 (12.6 + 46.9) / 59.5 (12.1 + 47.1) |
| splat model | — | 200 000 Gaussians, 3000 it (65.3 it/s), 90 views 600×800, memory plan 810 MB (GPU 301 MB) |
| registered / input keyframes | 90 / 90 | 90 / 90 |
| ATE / extent (×1e-4) | 2.88 / 2.87 | 2.88 / 2.89 |
| rotation error p50 (deg) | 0.056 / 0.052 | 0.056 / 0.056 |
| focal error (%) | 0.156 / 0.140 | 0.156 / 0.157 |
| scale error | -0.0037 / -0.0039 | -0.0037 / -0.0039 |
| gravity error (deg) | 0.021 / 0.021 | 0.021 / 0.021 |
| heading error (deg) | -0.45 / -0.45 | -0.45 / -0.45 |
| dense points | 568 k / 562 k | 1787 k / 1761 k |
| dense precision / completeness / F (run 1) | 0.998 / 0.996 / 0.997 | 1.000 / 0.997 / 0.998 |
| dense F (run 2) | 0.997 | 0.998 |
| dense accuracy p50 (mm) | 0.43 / 0.44 | 0.30 / 0.30 |
| enforced checks (run 1 / run 2) | pass / pass | pass / pass |

| yard-real, fast preset | before (`eb131af`) | after (`claude/breakthrough`) |
|---|---|---|
| host load (1-min mean, run 1 / run 2) | 5.0 / 3.8 | 5.0 / 4.4 |
| pipeline total (s, run 1 / run 2) | 193.9 / 180.0 | 152.2 / 146.9 (without splat 110.3 / 105.2) |
| · capture (live tracking of every frame) | 36.4 / 33.0 | 32.9 / 32.7 |
| · post-capture processing | 154.7 / 144.8 | 114.7 / 109.7 (without splat 72.8 / 68.1) |
| &nbsp;&nbsp;refine (run 1: matching + reconstruction) | 99.1 / 91.2 (34.3 + 64.4) | 27.4 / 23.2 (16.1 + 10.9) |
| &nbsp;&nbsp;densify (run 1: depth maps + fusion) | 47.1 / 45.5 (37.8 + 9.2) | 36.5 / 36.6 (27.6 + 8.9) |
| &nbsp;&nbsp;mesh | 7.8 / 7.2 | 8.0 / 7.2 |
| &nbsp;&nbsp;splat (prepare + training) | — | 41.9 (12.0 + 29.6) / 41.6 (12.1 + 29.3) |
| splat model | — | 200 000 Gaussians, 3000 it (104.6 it/s), 81 views 800×600, memory plan 758 MB (GPU 284 MB) |
| registered / input keyframes | 81 / 81 | 81 / 81 |
| ATE / extent (×1e-4) | 3.05 / 3.13 | 3.05 / 3.06 |
| rotation error p50 (deg) | 0.074 / 0.073 | 0.076 / 0.075 |
| focal error (%) | 0.160 / 0.156 | 0.162 / 0.162 |
| scale error | 0.0089 / 0.0094 | 0.0093 / 0.0098 |
| gravity error (deg) | 0.061 / 0.061 | 0.061 / 0.061 |
| heading error (deg) | -0.91 / -0.91 | -0.91 / -0.91 |
| dense points | 748 k / 752 k | 1760 k / 1837 k |
| dense precision / completeness / F (run 1) | 0.805 / 0.409 / 0.542 | 0.933 / 0.395 / 0.555 |
| dense F (run 2) | 0.540 | 0.557 |
| dense accuracy p50 (mm) | 25.01 / 25.02 | 8.96 / 8.78 |
| enforced checks (run 1 / run 2) | denseAccuracyMedian 0.0250 FAIL / denseAccuracyMedian 0.0250 FAIL | pass / pass |

* **Post-capture processing is 23 % faster on object-real (25 % on yard-real) with the splat stage included, and 52 %
  (53 %) faster without it** (object 202.9 → 156.1 s, or 96.5 s without splat; yard 149.8 → 112.2 / 70.4 s; means of
  the two runs). Refine is 3.4× faster (object 135 → 39 s) and 3.6–3.9× (yard 95 → 25 s). Dense depth maps are 27–28 %
  faster. Capture (live tracking) and mesh are unchanged.
* **Sparse accuracy is unchanged**: the same 90 / 81 keyframes register, and ATE, rotation, focal, scale, gravity and
  heading agree within run-to-run noise.
* **Dense**: 3.1× / 2.4× more points. Accuracy p50 improves from 0.43 to 0.30 mm (object) and from 25.0 to 8.9 mm
  (yard). The yard baseline fails its enforced dense-accuracy check (≤ 20 mm) in both runs, as it did in the earlier
  baseline (24.2 mm); after passes. Yard completeness drops slightly (0.409 → 0.395).
* **Splat stage**: 59.6 s (object) and 41.8 s (yard). About 12 s of that is preparation, mostly view decoding and upload
  on the engine thread. Training runs 3 000 iterations at 65 / 105 it/s (18.8 / 11.6 ms per iteration at full
  resolution). The memory plan peaks at 810 / 758 MB of the 1.2 GB budget.

### Novel views: held-out and off-trajectory

* **Held-out**: the 24 non-keyframes of the run (1–3 frames from a keyframe), with the default keyframe-relative pose.
  The **Sim3** row maps the same frames through the run's global Sim3, the mapping the off-trajectory sets have to use,
  so compare those against it.
* **Off-trajectory**: test datasets of the SAME seeded scene on shifted paths, 24 frames evenly spread over 200 / 240
  (`--test-dataset`, Sim3 pose). The object set shifts the orbit radius ±5 cm and the height +12 cm, plus a +25 cm
  stress case. The yard set shifts the walk 1.5 m outward / inward of the loop and its height +0.75 m, plus a +1.5 m
  stress case. All keep the -real artefacts but drop the three or four fast glance pans, whose frames are heavily
  blurred (the held-out set likewise skips the frames the capture rejected as blurred). "Nearest training camera" is the mean GT distance and optical-axis angle
  from each test view to the closest registered keyframe.
* **Scene identity check**: the geometry and textures depend only on the seed (yard: also on the loop's `center` /
  `semiAxes`, which `nvs.mjs` checks). Regenerating `object-real` and `yard-real` gave byte-identical JPEGs (200 / 200
  and 240 / 240). Every off-trajectory set has byte-identical `gt_points.bin` / normals. A yard regeneration with
  `lateralOffset = 1e-9` keeps 192 of 240 JPEGs byte-identical; the other 48 differ by at most 4 / 255 (72–87 dB).
* **Columns**: `full` is what the user sees, composited over black (holes count). `covered` uses each renderer's own
  pixels. `cc full` is after a per-view affine colour fit. `common` is every renderer on the pixels ALL of the run's
  renderers cover, roughly the mesh's footprint. The app's default view is `mesh-lit`, both before and after (the
  viewer on this branch does not show splats yet).

| test views (nearest training camera) | renderer | coverage | full PSNR / SSIM | covered PSNR / SSIM | cc full PSNR | common PSNR / SSIM (coverage) | shift px |
|---|---|---|---|---|---|---|---|
| held-out (1.8 cm / 2.0°) | before `mesh-lit` | 0.609 | 8.64 / 0.528 | 23.57 / 0.876 | 8.68 | 23.56 / 0.876 (0.61) | 0.12 |
|  | after `mesh-lit` | 0.610 | 8.65 / 0.531 | 23.55 / 0.877 | 8.68 | 23.55 / 0.877 (0.61) | 0.13 |
|  | after `mesh-unlit` | 0.610 | 8.71 / 0.552 | 28.65 / 0.915 | 8.72 | 28.66 / 0.915 (0.61) | 0.08 |
|  | after `splat` | 0.999 | **31.29 / 0.959** | 30.54 / 0.949 | 33.68 | 31.04 / 0.948 (0.61) | 0.08 |
| held-out (Sim3 pose) (1.8 cm / 2.0°) | before `mesh-lit` | 0.609 | 8.64 / 0.520 | 23.36 / 0.862 | 8.67 | 23.35 / 0.862 (0.61) | 0.49 |
|  | after `mesh-lit` | 0.610 | 8.64 / 0.522 | 23.34 / 0.863 | 8.67 | 23.34 / 0.863 (0.61) | 0.50 |
|  | after `mesh-unlit` | 0.610 | 8.70 / 0.543 | 28.02 / 0.900 | 8.71 | 28.03 / 0.900 (0.61) | 0.42 |
|  | after `splat` | 0.999 | **30.10 / 0.943** | 29.57 / 0.934 | 31.90 | 29.73 / 0.929 (0.61) | 0.40 |
| −5 cm radius (5.1 cm / 3.4°) | before `mesh-lit` | 0.636 | 8.96 / 0.531 | 22.78 / 0.844 | 9.00 | 22.77 / 0.843 (0.63) | 0.53 |
|  | after `mesh-lit` | 0.637 | 8.97 / 0.534 | 22.76 / 0.844 | 9.00 | 22.76 / 0.844 (0.64) | 0.53 |
|  | after `mesh-unlit` | 0.637 | 9.04 / 0.557 | 27.39 / 0.883 | 9.05 | 27.39 / 0.883 (0.64) | 0.46 |
|  | after `splat` | 1.000 | **30.20 / 0.937** | 29.43 / 0.928 | 31.84 | 29.25 / 0.920 (0.64) | 0.44 |
| +5 cm radius (5.0 cm / 2.8°) | before `mesh-lit` | 0.587 | 8.38 / 0.503 | 23.70 / 0.866 | 8.41 | 23.68 / 0.865 (0.58) | 0.46 |
|  | after `mesh-lit` | 0.588 | 8.38 / 0.505 | 23.68 / 0.867 | 8.41 | 23.68 / 0.867 (0.59) | 0.46 |
|  | after `mesh-unlit` | 0.588 | 8.43 / 0.524 | 28.22 / 0.903 | 8.44 | 28.22 / 0.903 (0.59) | 0.40 |
|  | after `splat` | 0.996 | **28.63 / 0.937** | 29.06 / 0.930 | 29.72 | 29.59 / 0.925 (0.59) | 0.41 |
| +12 cm height (11.0 cm / 10.0°) | before `mesh-lit` | 0.648 | 9.54 / 0.561 | 24.19 / 0.876 | 9.57 | 24.15 / 0.875 (0.64) | 0.45 |
|  | after `mesh-lit` | 0.648 | 9.55 / 0.564 | 24.15 / 0.876 | 9.58 | 24.15 / 0.876 (0.65) | 0.45 |
|  | after `mesh-unlit` | 0.648 | 9.61 / 0.581 | 28.31 / 0.906 | 9.62 | 28.32 / 0.907 (0.65) | 0.40 |
|  | after `splat` | 0.992 | **26.80 / 0.913** | 27.39 / 0.906 | 27.45 | 29.18 / 0.921 (0.65) | 0.46 |
| +25 cm height (stress) (23.7 cm / 19.3°) | before `mesh-lit` | 0.676 | 10.71 / 0.584 | 24.85 / 0.877 | 10.75 | 24.80 / 0.877 (0.66) | 0.41 |
|  | after `mesh-lit` | 0.678 | 10.73 / 0.587 | 24.81 / 0.878 | 10.76 | 24.80 / 0.878 (0.67) | 0.41 |
|  | after `mesh-unlit` | 0.678 | 10.79 / 0.602 | 28.19 / 0.901 | 10.80 | 28.21 / 0.902 (0.67) | 0.39 |
|  | after `splat` | 0.963 | **22.14 / 0.851** | 24.20 / 0.857 | 22.40 | 27.85 / 0.892 (0.67) | 0.47 |

| test views (nearest training camera) | renderer | coverage | full PSNR / SSIM | covered PSNR / SSIM | cc full PSNR | common PSNR / SSIM (coverage) | shift px |
|---|---|---|---|---|---|---|---|
| held-out (18.4 cm / 3.0°) | before `mesh-lit` | 0.644 | 9.75 / 0.370 | 23.16 / 0.625 | 9.82 | 23.19 / 0.626 (0.63) | 0.28 |
|  | after `mesh-lit` | 0.655 | 9.81 / 0.378 | 23.00 / 0.624 | 9.89 | 23.06 / 0.626 (0.64) | 0.42 |
|  | after `mesh-unlit` | 0.655 | 9.91 / 0.392 | 26.74 / 0.657 | 9.92 | 26.89 / 0.658 (0.64) | 0.38 |
|  | after `splat` | 1.000 | **31.07 / 0.872** | 30.37 / 0.863 | 32.30 | 31.27 / 0.851 (0.64) | 0.07 |
| held-out (Sim3 pose) (18.4 cm / 3.0°) | before `mesh-lit` | 0.644 | 9.75 / 0.370 | 23.05 / 0.622 | 9.82 | 23.09 / 0.623 (0.63) | 0.81 |
|  | after `mesh-lit` | 0.655 | 9.81 / 0.377 | 22.89 / 0.621 | 9.89 | 22.94 / 0.622 (0.64) | 0.94 |
|  | after `mesh-unlit` | 0.655 | 9.91 / 0.390 | 26.52 / 0.651 | 9.92 | 26.66 / 0.653 (0.64) | 1.02 |
|  | after `splat` | 1.000 | **29.22 / 0.818** | 28.80 / 0.812 | 30.06 | 29.66 / 0.789 (0.64) | 0.91 |
| 1.5 m outward (1.51 m / 4.1°) | before `mesh-lit` | 0.679 | 10.13 / 0.387 | 23.12 / 0.613 | 10.21 | 23.16 / 0.614 (0.67) | 1.08 |
|  | after `mesh-lit` | 0.694 | 10.22 / 0.396 | 22.92 / 0.611 | 10.30 | 22.99 / 0.613 (0.68) | 1.36 |
|  | after `mesh-unlit` | 0.694 | 10.33 / 0.409 | 26.20 / 0.639 | 10.34 | 26.36 / 0.641 (0.68) | 1.24 |
|  | after `splat` | 1.000 | **27.91 / 0.774** | 27.60 / 0.769 | 28.51 | 28.45 / 0.739 (0.68) | 1.09 |
| 1.5 m inward (1.51 m / 4.6°) | before `mesh-lit` | 0.486 | 8.92 / 0.273 | 22.56 / 0.625 | 8.97 | 22.59 / 0.626 (0.48) | 0.83 |
|  | after `mesh-lit` | 0.496 | 8.97 / 0.279 | 22.39 / 0.623 | 9.02 | 22.44 / 0.624 (0.49) | 0.94 |
|  | after `mesh-unlit` | 0.496 | 9.04 / 0.291 | 26.47 / 0.662 | 9.04 | 26.63 / 0.664 (0.49) | 0.86 |
|  | after `splat` | 0.999 | **25.32 / 0.796** | 25.82 / 0.796 | 25.58 | 29.98 / 0.806 (0.49) | 0.77 |
| +0.75 m height (76.7 cm / 3.3°) | before `mesh-lit` | 0.590 | 8.87 / 0.339 | 23.11 / 0.623 | 8.92 | 23.18 / 0.624 (0.58) | 0.95 |
|  | after `mesh-lit` | 0.602 | 8.92 / 0.346 | 22.94 / 0.620 | 8.97 | 23.02 / 0.622 (0.59) | 0.93 |
|  | after `mesh-unlit` | 0.602 | 8.99 / 0.358 | 26.39 / 0.650 | 8.99 | 26.64 / 0.653 (0.59) | 0.89 |
|  | after `splat` | 0.999 | **20.88 / 0.762** | 21.26 / 0.762 | 21.29 | 28.77 / 0.766 (0.59) | 0.80 |
| +1.5 m height (stress) (1.51 m / 3.3°) | before `mesh-lit` | 0.502 | 8.33 / 0.278 | 22.53 / 0.607 | 8.37 | 22.62 / 0.609 (0.49) | 1.07 |
|  | after `mesh-lit` | 0.514 | 8.37 / 0.285 | 22.32 / 0.604 | 8.42 | 22.42 / 0.606 (0.50) | 1.27 |
|  | after `mesh-unlit` | 0.514 | 8.42 / 0.295 | 25.11 / 0.634 | 8.43 | 25.38 / 0.637 (0.50) | 1.06 |
|  | after `splat` | 0.984 | **18.19 / 0.707** | 19.07 / 0.721 | 18.74 | 27.25 / 0.729 (0.50) | 0.89 |

* **The splat view wins everywhere on full frame.** On held-out frames it scores 31.3 / 31.1 dB (SSIM 0.959 / 0.872)
  against 8.6 / 9.8 dB for the app's lit mesh, because it fills the frame (coverage ≥ 0.96 vs 0.49–0.69). Off
  trajectory the margin over `mesh-lit` is still +9.8 to +21.2 dB.
* **Splats do extrapolate worse than they interpolate.** Against the held-out Sim3 row, the splat's full-frame PSNR is
  unchanged at −5 cm radius (+0.1 dB) and falls by 1.5 dB at +5 cm and 3.3 dB at +12 cm height (object). It falls by
  1.3 / 3.9 dB at 1.5 m outward / inward and 8.3 dB at +0.75 m height (yard). The stress cases lose 8.0 dB (object
  +25 cm, 24 cm / 19° from the nearest training camera) and 11.0 dB (yard +1.5 m). Height is the weak axis, because
  both captures hold a nearly constant height (object 10–30 cm above the object centre, yard 1.5 m ± 6 cm). From
  above, the tree crowns render as open bowls (their tops were never seen), the ground beyond the fence smears into a
  brown band, and the floor at the bookshelf streaks (see the views below). The geometry renderers barely move:
  `mesh-unlit` covered stays within 0.7 dB of its held-out Sim3 value (object) and 1.4 dB (yard +1.5 m).
* **On the mesh's own pixels the splat still leads.** On the `common` pixels the splat beats the unlit mesh by
  0.9–1.9 dB (object) and 1.9–3.4 dB (yard) off trajectory, and the lit app view by 3.0–7.5 dB. The one exception is
  object +25 cm, where it trails `mesh-unlit` by 0.4 dB. The off-trajectory loss is therefore mostly in the regions the
  mesh does not reconstruct at all (walls, sky, far ground, tree tops).
* **Before vs after mesh**: `mesh-lit` agrees within 0.21 dB and 0.015 coverage on every set. The denser cloud does not
  change the meshed view.
* **Pose-mapping sensitivity**: `--pose relative` composes with the keyframe nearest in space. It cuts the residual
  shift from 0.4–1.4 px to 0.1–0.6 px and raises the splat's full-frame PSNR by 0.1–1.3 dB (object +12 cm 27.27, +25 cm
  22.26; yard +0.75 m 21.10, +1.5 m 18.30, outward 29.23). The conclusions do not change.

### Balanced preset (object-real, one run each)

`--quality balanced` (after adds `--splat --nvs`). The dense views and splat training views are 720×960, the full
frame, so the novel views are scored at 720×960 and are not comparable with the fast rows above.

| object-real, balanced | before | after |
|---|---|---|
| host load (1-min mean) | 3.6 | 4.3 |
| pipeline total (s) | 390.5 | 405.9 (273.6 without splat) |
| post-capture processing (s) | 345.1 | 358.9 (226.6 without splat) |
| · refine / densify / mesh (s) | 136.0 / 173.4 / 34.4 | 41.7 / 147.6 / 36.0 |
| · splat (prepare + training, s) | — | 132.3 (17.4 + 114.6): 5 000 it at 44.5 it/s, 300 000 Gaussians, memory plan 1063 MB (GPU 439 MB) |
| ATE / extent, rot p50, focal, scale | 2.87e-4, 0.051°, 0.150 %, −0.0049 | 2.82e-4, 0.051°, 0.119 %, −0.0048 |
| dense points, P / C / F, accuracy p50 | 1403 k, 0.999 / 0.999 / 0.999, 0.41 mm | 3518 k, 1.000 / 0.999 / 0.999, 0.27 mm |
| held-out `mesh-lit` full / covered PSNR, coverage | 8.63 / 23.29 dB, 0.609 | 8.63 / 23.29 dB, 0.610 |
| held-out `splat` full PSNR / SSIM, covered, cc full | — | **31.71 / 0.956**, 31.26, 34.67 dB (coverage 1.000) |

At balanced, the splat stage (132 s) costs more than refine saves, so post-capture processing is 4 % slower with it
and 34 % faster without it. The 1063 MB memory plan leaves 137 MB of the 1.2 GB tab budget.

### iPhone estimate (not measured)

These are estimates scaled from the M4 means of the two runs, not measurements (measure them with
`run-e2e.mjs <dataset> --device phone --splat`, "On a real iPhone"). Each M4 time is multiplied by
peak ratio / sustained fraction, using the given M4 GPU ≈ 1.26× A19 Pro and 1.76× A18 Pro peak, sustained at
0.55–0.68 of peak: ×1.85–2.29 for an A19 Pro and ×2.59–3.20 for an A18 Pro. The factor is applied to every stage. That
is optimistic for the CPU-bound parts (refine's bundle adjustment on the CPU pool, view decoding and undistortion in
dense / splat prepare): they run on 6 phone cores instead of 10. Capture is live and paced by the recording, so it is
left out.

| stage | M4 (measured) | A19 Pro (est.) | A18 Pro (est.) |
|---|---|---|---|
| object-real post-capture processing, before | 202.9 s | 6.3–7.7 min | 8.8–10.8 min |
| object-real post-capture processing, after | 156.1 s | 4.8–6.0 min | 6.7–8.3 min |
| · of which splat stage (training) | 59.6 s (47.0 s) | 110–137 s (87–108 s) | 2.6–3.2 min (2.0–2.5 min) |
| · after, without the splat stage | 96.5 s | 3.0–3.7 min | 4.2–5.1 min |
| yard-real post-capture processing, before | 149.8 s | 4.6–5.7 min | 6.5–8.0 min |
| yard-real post-capture processing, after | 112.2 s | 3.5–4.3 min | 4.8–6.0 min |
| · of which splat stage (training) | 41.8 s (29.4 s) | 77–96 s (55–67 s) | 108–134 s (76–94 s) |
| splat iteration at full resolution (object / yard) | 18.8 / 11.6 ms | 35–43 / 21–26 ms | 49–60 / 30–37 ms |

Memory is not scaled. The fast splat plan peaks at 758–810 MB of the 1.2 GB tab budget on both scenes. Checkpoints
write 22 MB every 500 iterations (146–155 ms each here, in Chromium's IndexedDB); their cost on iOS is unmeasured.

### Side-by-side views

GT | splat | `mesh-lit` (app default) panels, 600×800 / 800×600. Views 5, 13 and 21 of the 24 are shown for the held-out
set and for the height-shifted set (object +12 cm, yard +0.75 m). They were made with `nvs.mjs … --frames <i,j,k>
--renderers splat,mesh-lit --save-images`, from the middle third (the render) of each `view-<frame>-<renderer>.png`
panel:

| set | frames | splat / `mesh-lit` full PSNR (dB) |
|---|---|---|
| object-real held-out | 31, 106, 172 | 29.75 / 10.06, 30.91 / 7.80, 31.54 / 7.59 |
| object-real +12 cm | 37, 104, 170 | 24.72 / 11.31, 27.21 / 9.08, 25.91 / 8.36 |
| yard-real held-out | 45, 125, 204 | 30.97 / 11.54, 33.09 / 10.62, 29.18 / 8.61 |
| yard-real +0.75 m | 45, 125, 205 | 18.47 / 10.92, 23.59 / 9.64, 19.24 / 8.06 |

On the +0.75 m yard views the splat shows open-bowl tree crowns and a smeared band above the fence where it never saw
the ground beyond. The mesh has the same bowl crowns and black holes for everything it did not reconstruct.

### Reproduce

```sh
# before: detached worktree of eb131af + the launch patch
git worktree add --detach ../sfm-bt-base eb131af && git -C ../sfm-bt-base cherry-pick bf55e6c
(cd ../sfm-bt-base && E2E_GPU=host node e2e/run-e2e.mjs object-real --engine worker --tag base-final)
E2E_GPU=host node e2e/nvs.mjs object-real ../sfm-bt-base/e2e/out/e2e/object-real-base-final --common
# after
E2E_GPU=host node e2e/run-e2e.mjs object-real --engine worker --splat --nvs --tag final
# off-trajectory test sets (same seed; the glance pans dropped; ~20–30 s each on the M4)
E2E_GPU=host node e2e/generate-synth.mjs object-real --name object-real-up12 --depth-every 0 --visibility-every 0 \
  --set 'trajectory.shake.pans=[]' --set trajectory.heightAbove=0.32    # in5 / out5: trajectory.radius=0.40 / 0.50
E2E_GPU=host node e2e/generate-synth.mjs yard-real --name yard-real-up75 --depth-every 0 --visibility-every 0 \
  --set 'trajectory.shake.pans=[]' --set trajectory.height=2.25    # in150 / out150: trajectory.lateralOffset=-1.5 / 1.5
E2E_GPU=host node e2e/nvs.mjs object-real e2e/out/e2e/object-real-final --test-dataset object-real-up12 --common \
  --renderers points,mesh-lit,mesh-unlit,splat                            # → <run>/nvs-object-real-up12/nvs.json
```

## Splat-stage experiments on a frozen session (`splat-replay.mjs`)

Re-runs only the engine's splat stage (`runSplat`, the function `PhotogrammetryEngine.splat` runs) on the inputs of
one e2e run, once per variant, and scores every variant on the held-out frames and the off-trajectory test sets.
SfM, dense and mesh are not repeated, so a variant costs one splat stage (≈ 40–60 s on the M4) plus ≈ 35 s per
test set; all variants of a scene share exactly the same poses, dense cloud and depth maps.

```sh
# once per scene: an e2e run that writes its session store to <run>/session/ after mesh (keyframe JPEGs + records,
# sparse model, dense cloud, fusion-filtered depth maps; ≈ 310–330 MB)
E2E_GPU=host node e2e/run-e2e.mjs object-real --engine worker --splat --tag gr-base --set persist=true --set dumpSession=true
# variants = SplatTrainerOptions overrides (EngineConfig.splat.trainer, on top of the preset's trainer defaults)
E2E_GPU=host node e2e/splat-replay.mjs e2e/out/e2e/object-real-gr-base --out e2e/out/replay/object \
  --variant before='{"colourCorrection":{},"geometry":false}' --variant ccoff='{"colourCorrection":false}'
#   --variants FILE (name → overrides JSON), --tests held,sim3,<test datasets> (default: held, sim3 and every
#   e2e/data/<dataset>-* set), --skip-nvs (train only), --nvs-only (score what is missing; resumes), --repeat N,
#   --splat-previews (every variant trains with SplatOptions.preview, as the app does)
# preview cost: two sequential invocations into the same --out, without and with previews (compare "stage s")
E2E_GPU=host node e2e/splat-replay.mjs e2e/out/e2e/object-real-gr-base --out e2e/out/replay/previews --variant off='{}' --skip-nvs
E2E_GPU=host node e2e/splat-replay.mjs e2e/out/e2e/object-real-gr-base --out e2e/out/replay/previews --variant on='{}' --splat-previews --skip-nvs
```

* **Faithfulness**: the page (`harness.html`, `window.replaySplat`, src/harness/engine/replay.ts) loads the dump into
  in-memory engine stores and calls `runSplat` with the run's quality and mode, a 1200 MB budget, the dense cloud
  and the depth maps; the trained model goes through the engine's stored f16 encoding and 'ply-splat' export. A
  replay of the unchanged stage reproduced the engine run: object-real held-out 31.32 dB (engine run 31.29),
  stage 59.8 s (59.7 s).
* **Output**: `<out>/<variant>/{splats.ply, result.json (copied from the run), replay.json, nvs*/nvs.json}` and
  `<out>/summary.json`; the script prints full-frame PSNR per test set, the off-trajectory mean and the stage times.
  `profile: true` in a variant adds per-pass GPU ms per iteration. With `--splat-previews`, `replay.json` holds the
  snapshots (`previews`: iteration, arrival ms, PSNR; `firstTrainingMs`) and the summary their count, first arrival,
  largest gap while training and PSNR first → last → final model.
* **Noise**: five replays of the unchanged stage spread 0.11 dB held-out (object 31.21–31.32, yard 31.07–31.14),
  up to 0.2 dB on single off-trajectory sets (yard +0.75 m: 20.48–20.68) and 0.05 (object) / 0.09 dB (yard) in the
  off-trajectory means (the GPU's float CAS adds make runs non-deterministic). Stage times are only comparable within a sequential run (see below): the
  quality sweeps ran two to four jobs at once.
* **Killed browsers**: on this host the installed Chrome was killed from outside every 10–20 min (both instances at
  once); the script times out the stuck evaluation (150 s per test set, 15 min per training) and retries.

### Geometry supervision, shape regularisers, colour correction (2026-10-20, M4 / Metal, fast preset)

Sessions: `object-real-gr-base` / `yard-real-gr-base` (this branch, worker engine, persistent session; 90 / 81
training views 600×800 / 800×600, 40 dense views with filtered depth maps each, 200 000 Gaussians, 3 000 iterations).
Every variant was trained on the same frozen inputs and scored with `nvs.mjs` (full-frame PSNR, splat renderer, 24
views per set): held-out (relative pose), held-out through the Sim3, the four off-trajectory sets of each scene
(Sim3 pose), and two **drift-free** sets per scene generated for this experiment (below). Variant names are
`SplatTrainerOptions` overrides; "until" = fraction of the iterations the geometry loss runs for (weights decay
0.5 → 0.05 for depth, 0.1 → 0.01 for normals over that fraction unless noted), "grazing 2" = target weight × cos²
between MVS normal and ray. Rows with several runs are means; the five "before" runs spread 31.21–31.32 dB held-out
(object) and 22.78–22.87 dB off-trajectory mean (yard).

| variant (runs) | object held / Sim3 | object −5 / +5 / +12 / +25 cm | object off mean (Δ) | yard held / Sim3 | yard in / out 1.5 m / +0.75 / +1.5 m | yard off mean (Δ) | off mean Δ both | drift-free object path / +12 cm, yard path / +0.75 m |
|---|---|---|---|---|---|---|---|---|
| before (CC on, no geometry) (5) | 31.28 / 30.19 | 30.27 / 28.72 / 26.99 / 22.43 | 27.10 (+0.00) | 31.10 / 29.23 | 25.08 / 27.95 / 20.59 / 17.72 | 22.83 (+0.00) | +0.00 | 32.53 / 27.81, 30.27 / 20.62 |
| geo: depth + normal, whole run (1) | 31.30 / 30.21 | 30.29 / 28.85 / 26.89 / 21.94 | 26.99 (-0.11) | 30.84 / 29.12 | 23.39 / 28.09 / 21.32 / 17.80 | 22.65 (-0.19) | -0.15 | — / —, — / — |
| depth only, whole run (1) | 31.30 / 30.20 | 30.34 / 28.76 / 26.81 / 21.82 | 26.93 (-0.17) | 30.89 / 29.15 | 23.55 / 28.10 / 21.53 / 18.38 | 22.89 (+0.06) | -0.06 | — / —, — / — |
| depth + normal, until 0.5 (1) | 31.33 / 30.23 | 30.36 / 28.82 / 27.11 / 22.15 | 27.11 (+0.01) | 30.99 / 29.20 | 24.78 / 28.07 / 21.36 / 18.51 | 23.18 (+0.35) | +0.18 | — / —, — / — |
| depth only, until 0.5 (1) | 31.31 / 30.24 | 30.36 / 28.86 / 27.02 / 22.13 | 27.09 (-0.01) | 31.04 / 29.21 | 24.82 / 28.07 / 21.40 / 18.54 | 23.21 (+0.37) | +0.18 | — / —, — / — |
| depth + normal, until 0.3 (1) | 31.27 / 30.18 | 30.30 / 28.80 / 27.05 / 22.25 | 27.10 (-0.00) | 30.99 / 29.18 | 25.32 / 28.02 / 21.28 / 18.64 | 23.31 (+0.48) | +0.24 | 32.56 / 27.87, 30.26 / 21.32 |
| **depth only, until 0.3 (new default)** (3) | 31.37 / 30.27 | 30.35 / 28.88 / 27.24 / 22.54 | 27.25 (+0.15) | 31.01 / 29.19 | 25.58 / 28.02 / 21.08 / 18.58 | 23.31 (+0.48) | +0.31 | 32.62 / 28.09, 30.25 / 21.12 |
| depth only, until 0.2 (1) | 31.30 / 30.23 | 30.31 / 28.77 / 27.05 / 22.24 | 27.09 (-0.01) | 30.91 / 29.11 | 25.61 / 27.98 / 21.00 / 18.53 | 23.28 (+0.44) | +0.22 | — / —, — / — |
| depth only 1 → 0.1, until 0.3 (1) | 31.42 / 30.31 | 30.37 / 28.86 / 27.31 / 22.49 | 27.26 (+0.16) | 30.95 / 29.19 | 25.30 / 28.01 / 21.00 / 18.52 | 23.21 (+0.37) | +0.26 | — / —, — / — |
| depth + normal 1 → 0.1 / 0.2 → 0.02, until 0.5 (1) | 31.27 / 30.16 | 30.33 / 28.96 / 26.96 / 22.04 | 27.07 (-0.03) | 30.89 / 29.18 | 25.29 / 28.09 / 21.46 / 18.47 | 23.33 (+0.49) | +0.23 | — / —, — / — |
| depth + normal, whole run, grazing 2 (1) | 31.27 / 30.20 | 30.32 / 28.75 / 27.00 / 22.42 | 27.12 (+0.02) | 30.94 / 29.16 | 24.53 / 28.06 / 21.30 / 17.98 | 22.97 (+0.13) | +0.08 | — / —, — / — |
| depth only, whole run, grazing 2 (1) | 31.32 / 30.24 | 30.36 / 28.83 / 26.85 / 22.18 | 27.06 (-0.05) | 31.02 / 29.20 | 24.76 / 28.02 / 21.45 / 17.99 | 23.06 (+0.22) | +0.09 | — / —, — / — |
| depth + normal, until 0.5, grazing 2 (1) | 31.29 / 30.22 | 30.27 / 28.75 / 26.81 / 22.02 | 26.96 (-0.14) | 31.04 / 29.20 | 25.09 / 28.00 / 21.42 / 18.51 | 23.25 (+0.42) | +0.14 | — / —, — / — |
| depth only, until 0.5, grazing 2 (1) | 31.37 / 30.27 | 30.33 / 28.81 / 26.76 / 21.85 | 26.94 (-0.16) | 31.04 / 29.19 | 25.27 / 28.00 / 21.26 / 18.36 | 23.22 (+0.39) | +0.11 | — / —, — / — |
| depth only 1 → 0.1, until 0.5, grazing 2 (1) | 31.34 / 30.21 | 30.32 / 28.85 / 27.36 / 22.35 | 27.22 (+0.12) | 31.00 / 29.22 | 25.08 / 28.06 / 21.22 / 18.52 | 23.22 (+0.39) | +0.25 | — / —, — / — |
| depth only, until 0.3, grazing 2 (1) | 31.32 / 30.19 | 30.31 / 28.71 / 26.81 / 21.99 | 26.95 (-0.15) | 31.03 / 29.18 | 25.49 / 27.93 / 21.22 / 18.65 | 23.32 (+0.49) | +0.17 | — / —, — / — |
| flatten 100 (1) | 31.35 / 30.26 | 30.33 / 28.80 / 26.95 / 22.13 | 27.05 (-0.05) | 31.13 / 29.24 | 25.01 / 27.97 / 20.62 / 17.76 | 22.84 (+0.01) | -0.02 | — / —, — / — |
| anisotropy cap 5, weight 1 (1) | 31.29 / 30.21 | 30.24 / 28.73 / 27.01 / 22.49 | 27.12 (+0.01) | 31.02 / 29.21 | 24.97 / 27.98 / 20.55 / 17.69 | 22.80 (-0.04) | -0.01 | — / —, — / — |
| colour correction off (3) | 33.76 / 32.11 | 31.42 / 29.84 / 27.72 / 22.90 | 27.97 (+0.87) | 31.92 / 29.77 | 25.09 / 28.07 / 20.41 / 17.53 | 22.78 (-0.06) | +0.40 | 30.72 / 27.40, 29.80 / 20.25 |
| colour correction pull 0.1 (1) | 31.37 / 30.32 | 30.38 / 28.76 / 27.03 / 22.37 | 27.14 (+0.04) | 31.16 / 29.26 | 25.20 / 27.98 / 20.67 / 17.80 | 22.91 (+0.08) | +0.06 | — / —, — / — |
| colour correction lr 5e-4 (1) | 31.98 / 30.78 | 30.72 / 29.27 / 27.10 / 22.20 | 27.32 (+0.22) | 31.32 / 29.39 | 25.16 / 28.02 / 20.54 / 17.71 | 22.86 (+0.02) | +0.12 | 32.44 / 27.66, 30.30 / 20.53 |
| CC off + depth only, until 0.3 (1) | 33.41 / 31.85 | 31.55 / 29.55 / 27.50 / 22.67 | 27.82 (+0.72) | 31.78 / 29.70 | 25.49 / 28.24 / 21.07 / 18.56 | 23.34 (+0.51) | +0.61 | 31.02 / 27.34, 29.87 / 20.99 |
| CC off + depth + normal, until 0.3 (1) | 33.30 / 31.80 | 31.59 / 29.64 / 27.60 / 22.77 | 27.90 (+0.80) | 31.80 / 29.71 | 25.29 / 28.27 / 21.29 / 18.67 | 23.38 (+0.54) | +0.67 | 30.93 / 27.42, 29.83 / 21.22 |
| CC off + depth + normal, until 0.5 (1) | 33.34 / 31.84 | 31.63 / 29.63 / 27.48 / 22.31 | 27.76 (+0.66) | 31.70 / 29.66 | 24.69 / 28.32 / 21.49 / 18.61 | 23.28 (+0.44) | +0.55 | — / —, — / — |
| CC off + depth only, until 0.5, grazing 2 (1) | 33.33 / 31.81 | 31.42 / 29.58 / 27.51 / 22.58 | 27.77 (+0.67) | 31.76 / 29.68 | 25.15 / 28.32 / 21.28 / 18.56 | 23.33 (+0.49) | +0.58 | — / —, — / — |

Drift-free sets (`--exposure-jitter 0 --wb-drift 0`, glance pans dropped, under `e2e/out/testsets/`):
`object-real-nx` / `yard-real-nx` follow the capture path, `object-real-up12-nx` / `yard-real-up75-nx` the +12 cm /
+0.75 m paths. They exist because the regular off-trajectory sets replay the capture's auto-exposure and
white-balance drift on the same clock (the drift is a seeded function of time, identical in every dataset of a seed),
so a test frame has the exposure of the capture frame taken at the same moment from a nearby place.

Splat-stage time, sequential runs on an otherwise idle GPU (host load 3–4), two runs each, interleaved:

| | object stage (prepare + train) | yard stage (prepare + train) |
|---|---|---|
| before | 59.1 / 59.7 s (13.0 + 46.3) | 40.0 / 40.3 s (11.6 + 28.5) |
| depth only, until 0.3 (new default) | 61.5 / 62.2 s (14.5 + 47.3): **+4.1 %** | 42.6 / 42.8 s (12.7 + 30.0): **+6.4 %** |
| depth + normal, until 0.3 | 61.6 s (13.2 + 48.3) | 42.8 s (12.4 + 30.2) |
| depth + normal, whole run (alone on the GPU) | 74.4 s (15.7 + 58.5): +24 % | |

Per-pass GPU time per iteration over the whole schedule (`profile: true`, object): depth + normal for the whole run
adds 2.9 ms to the rasteriser backward (7.61 → 10.49 ms: 13 instead of 9 accumulators and their CAS adds), 0.45 ms to
the loss (geometry kernel) and 0.17 ms to the rasteriser. With `until` 0.3 the geometry kernels only run during the
¼- and ½-resolution part, and the depth targets cost 1.0–1.5 s of preparation (40 maps resampled on the engine
thread) plus the extra pipelines.

* **Depth supervision helps where the MVS sees what the off-trajectory views see, and only when it stops early.** With
  the new default (inverse depth only, 0.5 → 0.05 over the first 30 %) the yard gains +0.48 dB off trajectory
  (+0.75 m +0.49, +1.5 m +0.86, inward +0.50) and +0.50 dB on the drift-free +0.75 m set; the object +0.15 dB (+12 cm
  +0.25, +25 cm +0.11, drift-free +12 cm +0.28). Held-out moves +0.09 (object) / −0.09 dB (yard), the Sim3 held-out
  +0.08 / −0.04, colour-fit held-out +0.2–0.3 dB. All within the rule (mean off-trajectory +0.31 dB, held-out ≥ −0.09
  dB, time +4–6 %): **on by default** (`SPLAT_TRAINER_DEFAULTS.geometry`).
* **Supervising the whole run punches holes.** Off-trajectory views that see ground the capture only saw at grazing
  angles (yard inward 1.5 m: the near ground) lose 1.5–1.7 dB: the depth loss pins the ground Gaussians onto a thin
  layer whose gaps only show from the new viewpoint (coverage 0.981 instead of 0.9995; on the covered pixels the
  render is as good as before). Stopping at 30–50 % lets the photometric phase close them (inward −1.7 → +0.5 dB);
  down-weighting grazing targets (cos²) cuts that loss to a third for whole-run supervision but adds nothing once it
  stops early.
* **Normals add nothing here.** The normal L1 against the MVS normals (DN-Splatter-style, rendered normal = smallest
  axis) gives the same off-trajectory gains as depth alone (until 0.3: +0.24 vs +0.31 dB) and needs the 13-accumulator
  kernels, so its weight defaults to 0 (depth-only kernels). Stronger depth weights (1 → 0.1) or until 0.2 are within
  noise of the default.
* **Flatter Gaussians do not move novel-view PSNR.** Both regularisers do what they should (object: Gaussians with
  mid/min > 5 73.5 → 99.7 % at flatten 100; needles max/mid > 5 8.3 → 0.1 % with the cap) but every set stays within
  ±0.07 dB: off: the smears and the bowl-shaped tree tops come from surfaces no training view saw, not from the shape
  of the Gaussians.
* **Colour correction stays on.** Without it, raw held-out PSNR rises by 2.5 (object) / 0.8 dB (yard) and the regular
  off-trajectory mean by +0.40 dB, but on the drift-free sets it LOSES 1.81 / 0.41 dB (object path / +12 cm) and 0.47 /
  0.37 dB (yard path / +0.75 m): without the per-view matrices the model bakes the capture's exposure into region-wise
  colours, which the regular test frames — exposed on the same clock — happen to reproduce. With the matrices the
  splats hold the mean appearance, which is what a viewer from a new place should see. The colour-fit PSNR is mixed
  (object +0.2, yard −0.1 to −0.3 dB without correction). A lower correction rate (lr 5e-4) or a stronger pull to
  identity (0.1) sit in between and gain nothing on the drift-free sets. Combining "correction off" with depth
  supervision also costs 0.35–0.45 dB object held-out against "correction off" alone.
* **What remains**: the height failures are content the capture never saw (object top and floor behind it at +25 cm,
  tree crowns and the ground behind the fence from +1.5 m). Depth supervision closes ≈ 0.5–0.9 dB of the yard's
  8.3–11 dB height gap; it cannot invent unseen surfaces.

End-to-end check with the new default (`E2E_GPU=host node e2e/run-e2e.mjs object-real --engine worker --splat --tag
gr-final`: a fresh non-persistent run, depth maps from the scratch IndexedDB store): "geometry targets for 40 of 90
views (1/2 size) in 1114 ms", splat stage 60.3 s (prepare 12.4 + train 47.7 s at 64.1 it/s), memory plan 863 MB
(GPU 353 MB; 811 / 301 MB in the "Splat stage in the engine" run), every enforced check passes. Splat full-frame PSNR:
held-out 31.30, −5 / +5 / +12 / +25 cm 30.47 / 28.91 / 27.16 / 22.40 dB (that section's run: 31.29; 30.20 / 28.63 /
26.80 / 22.14 dB).

Reproduce (after the two `gr-base` runs above; ≈ 5 min per variant and scene with all six sets):

```sh
E2E_GPU=host node e2e/generate-synth.mjs object-real --name object-real-up12-nx --out e2e/out/testsets --depth-every 0 \
  --visibility-every 0 --set exposureJitter=0 --set whiteBalanceDrift=0 --set 'trajectory.shake.pans=[]' --set trajectory.heightAbove=0.32
E2E_GPU=host node e2e/splat-replay.mjs e2e/out/e2e/yard-real-gr-base --out e2e/out/replay/yard \
  --variant before='{"geometry":false}' --variant default='{}' --variant normals='{"geometry":{"normal":[0.1,0.01]}}' \
  --variant whole-run='{"geometry":{"until":1}}' --variant ccoff='{"colourCorrection":false}' \
  --variant flat='{"flatten":100}' --variant aniso='{"anisotropy":{"maxRatio":5,"weight":1}}'
E2E_GPU=host node e2e/nvs.mjs yard-real e2e/out/replay/yard/default --renderers splat --test-dataset e2e/out/testsets/yard-real-up75-nx
```
