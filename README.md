# 3D Scan — on-device photogrammetry for the mobile web

A Progressive Web App that turns a continuous walk-around with an iPhone into a 3D model,
**entirely on the phone**: no upload, no server. Point the camera at an object on a table, or walk
through a backyard and the houses around it. The app picks sharp, informative frames automatically,
shows the reconstruction growing live, and produces a metric, gravity-aligned and (with GPS)
geo-referenced point cloud and mesh that you can measure and export.

* **Capture**: continuous video from the fixed wide lens. Keyframes are chosen by parallax, overlap,
  blur and gyro angular speed, with guidance shown while capturing ("move slower", "translate, don't
  just rotate", "tracking lost").
* **Sparse reconstruction**: RootSIFT features and matching on WebGPU, gyro-guided matching,
  COLMAP-style incremental structure-from-motion, robust bundle adjustment with self-calibration
  (focal, radial k1/k2, principal point), and vocabulary-tree loop closure.
* **Sensor fusion**:
  * Accelerometer: **metric scale** and gravity alignment, with the camera–IMU time offset estimated
    automatically.
  * Gyroscope: rotation priors and rolling-shutter / motion handling.
  * Compass: **true north**, with WMM2025 declination applied.
  * GPS: scale and geo-referencing for large areas.
* **Dense reconstruction**: PatchMatch multi-view stereo on WebGPU (ACMH/ACMM-style, inverse-depth
  sampling for scenes from 0.2 m to 100 m), consistency-checked depth-map fusion, halo and sky
  filtering, TSDF meshing with marching cubes, and mesh simplification.
* **Viewer**: three.js splat renderer with eye-dome lighting, mesh view, camera frustums, and a
  tap-to-measure tool (metric when scale is known).
* **Exports**: PLY (points and mesh), GLB, and a COLMAP text model with images in a ZIP. Exports go
  through the iOS share sheet.
* **Robust on iPhone**:
  * Memory planner for the tab's ~1.2 GB limit, with resume after the tab is killed.
  * One GPU command buffer in flight at a time, GPU-loss recovery, and CPU fallbacks.
  * Sessions persist in IndexedDB.
  * On-device benchmark and diagnostics.

## Using it on an iPhone

Requirements: **iOS / iPadOS 26 or newer** (WebGPU is on by default since Safari 26). The app runs
without WebGPU, slower, using CPU fallbacks. It must be served over **HTTPS**: camera, motion sensors
and geolocation need a secure context.

1. Open the deployed URL in Safari and use **Share → Add to Home Screen**. Storage of an installed
   web app is persistent; a Safari tab's storage can be evicted after 7 days without use.
2. Tap **New scan**, then choose:
   * **Object**: move slowly around it at a roughly constant distance, keeping ~70 % overlap. Cover
     it from 2–3 heights.
   * **Large area**: walk slowly and sidestep rather than only turning on the spot. Look at
     structures from several places and come back to the start (loop closure). GPS adds scale and
     geo-referencing.
   * **Auto**: lets the app decide.
   * **Quality**: *Fast / Balanced / Max* trades processing time and memory for resolution.
3. Tap **Start**. iOS asks for motion access, then camera, then location. Follow the on-screen
   guidance, and tap **Finish** when the live model covers what you need.
4. Processing builds the sparse model and aligns it to gravity, scale and north. In the viewer you
   can **Build dense cloud** and **Build mesh**, measure, and **Export**.

The **Diagnostics** screen shows WebGPU limits, the camera list, IMU rate and sign checks, and
storage. It also has a **Run benchmark** button that times GPU SIFT, matching, PatchMatch and bundle
adjustment on the device and estimates scan times per quality preset.

## Development

```sh
npm install
npm run dev          # Vite dev server (desktop: http://localhost:5173; ?mock=1 = mock engine)
npm run build        # typecheck + production build into dist/
npm test             # unit tests (Node)
npm run test:gpu     # WebGPU / DOM tests in headless Chromium (SwiftShader)
```

To test on a phone you need HTTPS. Either:

* push to `main` with GitHub Pages enabled (Settings → Pages → Source: *GitHub Actions*); the
  workflow in `.github/workflows/pages.yml` deploys `dist/`, or
* run `npm run dev` behind any HTTPS tunnel.

CI (`.github/workflows/ci.yml`) runs typecheck, unit tests and the build.

### End-to-end evaluation with ground truth

The pipeline is validated headlessly on rendered datasets with exact ground truth: poses,
intrinsics, 60 Hz IMU, GPS, compass and dense surface samples. There are clean variants and
phone-realistic variants with rolling shutter, focus breathing, hand shake, exposure drift and
JPEG compression.

```sh
npm run synth -- all                 # render e2e/data/{object,yard} (+ object-real, yard-real presets)
npm run e2e -- object --engine worker # replay through the real engine worker, score vs ground truth
npm run e2e:dense-gt -- all          # dense MVS alone, with ground-truth poses
npm run ui:smoke                     # full UI flow with the mock engine at iPhone size
```

See [`e2e/README.md`](e2e/README.md) for options, dataset layout, metrics and recorded results.

Results with the real engine worker on the clean synthetic datasets (SwiftShader, 480 px dense):

| | object (0.5 m orbit, 200 frames) | backyard (40 m walk, 240 frames) |
|---|---|---|
| registered keyframes | 100 % | 100 % |
| camera position error (ATE) | 0.29 mm (2e-4 of scene) | 8 mm (2e-4 of scene) |
| rotation error | 0.05° | 0.06° |
| focal length error | 0.03 % | 0.07 % |
| metric scale error | 0.4 % (IMU) | 0.01 % (IMU + GPS) |
| gravity / north error | 0.01° / 0.4° | 0.01° / 0.6° |
| dense F-score | 0.998 (τ = 13.6 mm) | 0.55 (τ = 0.1 m; far field limited by resolution) |

## How it works

```
UI thread                                    engine worker (module worker, WebGPU)
─────────                                    ─────────────────────────────────────
camera (fixed wide lens) ── FramePump ─────▶ frame ingest (orientation, blur/motion gate)
IMU · compass · GPS ───────── per frame ───▶ GPU RootSIFT ─ gyro-guided matching ─ keyframe policy
                                              │ keyframe: JPEG → IndexedDB, match vs recent +
three.js viewer ◀──── live sparse model ─────┤ retrieval candidates, LiveMapper (PnP, triangulate,
                                              │ local BA)
UI ◀──────── progress / results ─────────────┤ refine: loop-closure pairs → incremental SfM →
                                              │ self-calibrating BA → IMU/GPS/compass alignment
                                              │ dense: undistort → PatchMatch (photometric +
                                              │ geometric) → fusion → filters → TSDF mesh
                                              └ exports: PLY · GLB · COLMAP ZIP
```

* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): conventions, module contracts, and per-module
  design notes and review findings.
* [`docs/ENGINE.md`](docs/ENGINE.md): engine API, worker protocol and UI flow.
* [`docs/research/`](docs/research): verified iOS Safari facts on WebGPU, camera, sensors, memory
  and storage.

| Directory | Contents |
|---|---|
| `src/math`, `src/geometry` | linear algebra, minimal solvers (5-pt, 7/8-pt, homography, P3P/EPnP), LO-MSAC, two-view geometry |
| `src/ba` | Levenberg–Marquardt bundle adjustment with Schur complement, dense/PCG solvers, robust losses, priors |
| `src/features` | SIFT (CPU + WASM SIMD reference, WebGPU), GPU/CPU matching, guided matching |
| `src/sfm` | feature database, correspondence graph, incremental + live mapper, keyframing, retrieval, inertial / GPS / world alignment |
| `src/dense` | PatchMatch MVS (WebGPU + CPU), view selection, fusion, halo/sky filters, TSDF, marching cubes, simplification |
| `src/sensors` | camera capture, IMU normalisation (iOS sign fixes), compass + WMM2025, GPS, frame pump, permissions |
| `src/engine`, `src/workers` | the photogrammetry engine, memory planner, persistence, worker RPC |
| `src/app`, `src/viewer`, `src/io` | mobile UI, three.js viewer, storage and exports |
| `src/harness`, `e2e` | synthetic dataset generator, harnesses, metrics, UI flows |

## Limitations

* Device performance and memory budgets are derived from WebKit sources and documentation. They
  are **not yet measured on real iPhones**; the in-app benchmark exists to check them.
* Focus and exposure cannot be locked from the web. The pipeline self-calibrates and tolerates
  focus breathing, but strongly changing focus distance still costs some accuracy.
* Textureless, reflective or transparent surfaces and moving objects reconstruct poorly, as with
  any photogrammetry.
* For very large areas, dense completeness of distant structure is limited by image resolution
  and the number of views. Use the *Max* preset and capture structures from closer as well.
