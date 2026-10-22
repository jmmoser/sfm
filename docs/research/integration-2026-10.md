# Integration of the October 2026 roadmap slices (2026-10-22)

Measurements from 2026-10-22 on the integrated branch `claude/next-level` (merge of b2377c7 and the
commits listed below), headless Chromium 141 on SwiftShader, 4 shared cores, CPU features
(`engineConfig={"featureBackend":"cpu"}`). The code may have changed since — check the source before
relying on a number. Per-feature measurements are in their own reports (`imu-scale.md`,
`extend-scan.md`, `real-capture-robustness.md` "Phone-rate capture", `dense-tuning.md` "CPU/GPU
overlap", `splat-tuning.md`) and in the commit messages.

## What was merged

Coverage ring + guidance gate + too-dark guidance, run journal (interrupted-run card, light-mode
retry), extend scan ("Add more views"), raw capture recorder with verbatim replay and GT-free
scoring, object box (support plane + TSDF bounds), photoreal preview during training + "Use it
now", USDZ with a colour atlas + "View in AR", fly-through MP4 + SPZ export, phone bridge + nightly
CI tier, IMU metric scale at phone keyframe rates (interval estimator), dense CPU/GPU overlap.
The USDZ pivot uses the object box's bottom centre (`selectArPivot`, `objectRoiPivot`) when the
mesh has one.

## Test suites

| suite | result |
|---|---|
| `npm run typecheck`, `npm run build` | clean; no `__e2e` bridge code in `dist/` |
| unit (`npx vitest run --project unit`) | 190 files, 1793 tests pass (base aeff8f6: 166 / 1530) |
| GPU (`npx vitest run --project gpu`) | 299 pass, 2 skipped, 1 fail: `viewer.gpu.test.ts` "measures distances…" timing assertion (3.6 s vs 2 s) under full-suite load; passes alone (28 / 28) |
| `e2e/ui/smoke.mjs`, `layouts.mjs`, `recorder.mjs`, `flythrough.mjs --probe` | pass (recorder failed once under load, passed alone; no H.264 encoder on SwiftShader, so the video probe reports none) |
| `e2e/ui/states.mjs` | every scenario passes after the mock-engine fix below |

## Engine end to end (fast preset, dense at 480 px)

| | object-real | yard-real |
|---|---|---|
| registered / ATE ÷ extent / rotation p50 | 89 / 89, 2.73e-4, 0.049° | 81 / 81, 5.29e-4, 0.102° |
| scale error (source) / gravity / heading | +0.27 % (IMU) / 0.02° / −0.74° | +1.13 % (IMU + GPS) / 0.04° / −0.82° |
| dense accuracy p50 / completeness / F | 0.33 mm, 0.993, 0.996 | 9.5 mm, 0.344, 0.506 |
| object box | plane tilt 0.03°, confidence 0.94; IoU with `meta.focus` 0.72, contains focus − 1 cm | off (scene mode) |
| live coverage at the end | 50 %: mid band 12 / 12, high 0 / 12, prompt "look down" | not shown (scene) |
| exports | ply-points 42.4 MB, ply-mesh 2.2 MB, GLB 2.5 MB, COLMAP 19.1 MB, USDZ 8.8 MB | USDZ 10.8 MB |
| enforced checks | all pass | all pass |

The object-real capture never rises above 40° elevation, so the coverage ring correctly reports the
top as missing (the off-trajectory height loss in e2e/README "Novel views").

Object box, same model with and without it (`--no-object-roi`), mesh vertices scored against
`meta.focus` ± 2 cm (`e2e/metrics.mjs --focus-points`):

| object-real mesh | without box | with box |
|---|---|---|
| vertices / triangles | 242 637 / 472 004 | 43 263 / 84 448 |
| vertices outside the object | 73.7 % | 0 % |
| completeness of the object's GT samples (τ 13.6 mm) | 0.9975 | 0.9968 |

## Splat stage (object-real, 300 iterations, 20 k Gaussians, 240 px, 30 views, `--splat-previews`)

* 12 previews; the first 11.7 s into the stage (the first training report came at 31.6 s); PSNR
  of the preview against its training photo 15.3 → 22.1 dB (final model 22.3 dB). The largest gap
  between previews while training was 96.5 s: SwiftShader trains at 0.4 it/s.
* Exports: splat PLY 1.30 MB, SPZ 0.33 MB (3.9× smaller).
* Held-out views (`nvs.mjs`, the same 24 frames): PLY 24.44 dB / SSIM 0.832, the decoded SPZ
  24.42 dB / 0.831. With dense at 240 px the dense-completeness check fails (0.952 vs ≥ 0.98, tuned
  for 480 px); every other check passes.

## Unread frame grabs (e2e/ui coverage screenshot hang)

`e2e/ui/states.mjs coverage` passed every functional check but its last screenshot timed out, and
the scenario left behind slowed the next one. Cause, measured with `e2e/ui/gpu-backpressure.mjs`
(this machine, 10 s per mode):

| grab at 10 Hz from the fake camera `<video>` | grabs/s | GPU process RSS | screenshot right after |
|---|---|---|---|
| `createImageBitmap` + read through a 1×1 canvas + close | 4.4 | 277 MB (+1.8 MB/s) | 0.3 s |
| `createImageBitmap` + close, unread | 10.0 | 2 896 MB (+232 MB/s) | 15.3 s |

Chromium resolves `createImageBitmap(<video>)` before the GPU process has made the copy, so a
consumer that closes the bitmap unread never waits and the copies queue without bound (5–8 GB in
the longer runs, then a GPU-process crash); compositor readbacks (page screenshots) wait behind
them, while a 1×1 WebGL `readPixels` still returned in milliseconds. The MockEngine closed frames
unread; the real engine reads every frame, so production capture is unaffected. The MockEngine now
reads every frame (src/app/mock-engine.ts "Frame pixels"), the scenario keeps the 3D thumbnail
(hiding it had removed the only thing that slowed the pump), and `states.mjs` closes a scenario's
pages even when it fails.

## Phone-rate capture (final head d11c210, sparse, `--no-dense`)

`--stride 3` replays about 2 analysed frames per second, the rate an iPhone is expected to reach;
the "before" row is the 2026-10-21 run of aeff8f6 + the shared convergence solve (tag `s3-base`).
Every A/B run of the two slices behind it is in `imu-scale.md` and `real-capture-robustness.md`
"Phone-rate capture".

| yard-real | live registered / lost frames | refine | ATE ÷ extent, rot p50 | scale error (method) | checks |
|---|---|---|---|---|---|
| stride 3, before | 19 / 48, 38 of 80 lost | offline SfM (29 pending) | 5.10e-4, 0.111° | −6.4 % (GPS: IMU excitation 0.010 m/s²) | scaleError fails |
| stride 3, final | 48 / 48, 0 lost | live model kept | 5.32e-4, 0.119° | −0.9 % (IMU + GPS, interval estimator) | all pass |
| stride 1, final | 81 / 81, 0 lost | live model kept | 5.68e-4, 0.108° | +1.2 % (IMU + GPS, band-pass) | all pass |

object-real final: stride 3 50 / 50, 5.95e-4; stride 1 89 / 89, 2.72e-4; all checks pass. The cost
of re-triangulating neighbours is visible at full rate on yard-real (ATE ≈ +8 % over 4.8–5.3e-4,
far inside the 1e-3 limit); gyro-aided registration runs in refine's offline SfM only by default.

## Not measured

Nothing here ran on an iPhone or a real GPU: device timings, memory peaks, AR Quick Look,
WebCodecs H.264 and the run journal's kill classification still need the phone bridge
(e2e/README "On a real iPhone"). Dense CPU/GPU overlap is bit-identical, but its wall-time gain
cannot show on SwiftShader, where the "GPU" shares the CPU cores.
