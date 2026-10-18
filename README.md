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
  * Memory planner that keeps the tab within ≤ 1.2 GB (iOS kills it at ~1.5–2 GB), with resume
    after the tab is killed.
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

**Camera timing** has two probes. **Quick** (15 s) is for orientation: point at something textured and turn the phone to both landscape sides and back while it runs. **Still** (60 s after a 5 s warm-up, phone held still) is for drift. Each reports:
* how much rVFC `captureTime` jitters and drifts against the `MediaStreamTrackProcessor` capture clock, away from rotations;
* what happens around rotations: dropped frames, wrongly oriented frames, and the worst timestamp error;
* whether the processor's buffers stay sensor-landscape, and how a `VideoFrame` of the `<video>` is really rotated at each screen angle (thumbnail match against the upright `<video>`; size alone cannot tell at landscape angles);
* how long `createImageBitmap(<video>)` (full size and downscaled) and `new VideoFrame(<video>)` block the main thread.

This is the data for deciding whether a different capture path is worth building.

**VideoFrame grab (experimental)**, a switch under *Options* on the New scan screen, is the change
those measurements pointed to. Grabbing a frame with `createImageBitmap(<video>)` blocks the main
thread for about 15 ms on an iPhone, about 15% of it at the 10 grabs per second the app makes. With
the switch on, the app takes a `VideoFrame` of the `<video>` instead, which costs 0–3 ms. The engine
worker converts it, downscaled, and closes it straight away. On iOS these frames stay in the sensor's
landscape orientation, so each one is marked raw for the engine; the mark applies to that frame
only and is never saved with the session. The debug overlay shows the main-thread grab time and the
worker's conversion time.

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

Scans (photos, models, GPS fixes) live in the origin's IndexedDB. On a shared origin such as
`<user>.github.io/<repo>/`, every other project of that user can read them, so give a real deployment its
own origin (custom domain). The production build ships only the app (the e2e harness pages are dev-server
only), with a Content-Security-Policy and a versioned service worker (offline after the first visit).

CI (`.github/workflows/ci.yml`) runs typecheck, unit tests and the build.

### End-to-end evaluation with ground truth

The pipeline is validated headlessly on rendered datasets with exact ground truth: poses,
intrinsics, 60 Hz IMU, GPS, compass and dense surface samples. There are clean variants and
phone-realistic variants with rolling shutter, focus breathing, hand shake, exposure drift and
JPEG compression.

```sh
npm run synth -- all                 # render e2e/data/{object,yard}; `-- real`: object-real + yard-real
npm run e2e -- object --engine worker # replay through the real engine worker, score vs ground truth
npm run e2e:dense-gt -- all          # dense MVS alone, with ground-truth poses
npm run ui:smoke                     # full UI flow with the mock engine at iPhone size
node e2e/ui/states.mjs               # UI edge states; also e2e/ui/layouts.mjs (sizes), bench.mjs
```

See [`e2e/README.md`](e2e/README.md) for options, dataset layout, metrics and recorded results.

Results with the real engine worker: fast preset, all sensors, 480 px dense images
(`--dense-size 480`), and CPU features on SwiftShader (`--set 'engineConfig={"featureBackend":"cpu"}'`;
the WebGPU kernels give the same features but take 25 s per frame there). Final verification runs
(2026-10-17; sparse rows: `--no-dense`, tag `verify`; *-real* dense: tag `verify-dense480`; † clean
dense: the earlier `final` runs in e2e/README.md). Every `e2e/thresholds.json` check passes. The
*-real* datasets add rolling shutter, focus breathing, hand shake with fast pans, exposure /
white-balance drift and JPEG 0.85:

| | object (0.5 m orbit, 200 frames) | backyard (40 m walk, 240 frames) | object-real | yard-real |
|---|---|---|---|---|
| registered keyframes | 92 / 92 | 80 / 80 | 89 / 89 | 81 / 81 |
| camera position error (ATE / scene extent) | 2.1e-4 (0.29 mm) | 2.2e-4 (8.5 mm) | 2.8e-4 (0.39 mm) | 3.2e-4 (12 mm) |
| rotation error (median) | 0.08° | 0.05° | 0.08° | 0.09° |
| focal length error | +0.04 % | +0.07 % | +0.09 % | +0.16 % |
| metric scale error (source) | +0.3 % (IMU) | +0.1 % (IMU + GPS) | −0.9 % (IMU) | +1.6 % (IMU + GPS) |
| gravity / north error | 0.01° / −0.8° | 0.01° / −0.6° | 0.02° / −0.5° | 0.06° / −0.9° |
| dense precision / completeness (τ) | 0.9997 / 0.996 (13.6 mm)† | 0.964 / 0.379 (0.1 m)† | 0.9995 / 0.994 (13.6 mm) | 0.961 / 0.351 (0.1 m) |
| dense F-score / median accuracy | 0.998 / 0.24 mm† | 0.55 / 6.7 mm† | 0.997 / 0.31 mm | 0.51 / 8.8 mm |

Backyard completeness is limited by image resolution in the far field (0.23–0.34 even with exact poses).

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
| `src/sfm` | feature database, correspondence graph, incremental + live mapper, keyframing, retrieval, gyro checks, rolling shutter, inertial / GPS / world alignment |
| `src/dense` | PatchMatch MVS (WebGPU + CPU), view selection, fusion, halo/sky filters, TSDF, marching cubes, simplification |
| `src/sensors` | camera capture, IMU normalisation (iOS sign fixes), compass + WMM2025, GPS, frame pump, permissions |
| `src/engine`, `src/workers` | the photogrammetry engine, memory planner, persistence, worker RPC |
| `src/core`, `src/gpu` | shared types, WebGPU device acquisition |
| `src/app`, `src/viewer`, `src/io` | mobile UI (incl. on-device benchmark), three.js viewer, storage and exports |
| `public`, `tools` | service worker, web manifest, icons (and their generator) |
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
