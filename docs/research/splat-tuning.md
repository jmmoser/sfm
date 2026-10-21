# Gaussian-splat tuning

Measurements from October 2026 (2026-10 to 2026-10-20); the code may have changed since — check the source before relying on a number.

Everything here ran on an Apple M4 GPU (Metal, headless Chrome, `E2E_GPU=host`) on the synthetic
`object-real` / `yard-real` datasets; there is no iPhone measurement yet. Full per-view tables, commands
and ablations are in `e2e/README.md` (sections named below). Code: `src/splat/` (trainer, renderer),
`src/engine/splat.ts` (engine stage), `src/viewer/splat-layer.ts` (viewer).

## Trainer with ground-truth poses

`e2e/run-splat-gt.mjs` (e2e/README.md "Splat results"): exact poses and intrinsics, held-out frames,
300 k Gaussians max, initialisation = fused MVS cloud + sparse points (+ a 20 k sky shell on the yard).
Full-frame PSNR in dB:

| | 0 it | 1 000 | 3 000 | 7 000 | colour-fit @ 7k | SSIM @ 7k | train 3k / 7k |
|---|---|---|---|---|---|---|---|
| object-real 600×800 | 19.1 | 28.5 | 30.4 | 30.9 | 33.3 | 0.948 | 48 s / 177 s |
| object-real 720×960 | 18.9 | 27.9 | 29.7 | 30.1 | 32.0 | 0.930 | 50 s / 160 s |
| yard-real 800×600 (+ sky shell) | 17.2 | 27.9 | 29.5 | 29.9 | 30.9 | 0.843 | 23 s / 73 s |
| yard-real 960×720 (+ sky shell) | 17.0 | 27.1 | 28.5 | 28.9 | 29.7 | 0.803 | 27 s / 87 s |

For comparison, the app's other renderings of object-real held-out views: lit TSDF mesh 9.2 dB full frame (61 %
coverage), points + EDL 15.8 dB on covered pixels.

Ablations at 3 000 iterations:

* MVS seeding: +10 dB over random initialisation (20.2 dB), +0.6 dB over GT surface samples.
* 500 k Gaussians: +0.2 dB. SH degree 1: no gain on these Lambertian scenes (+8 % memory) — SH 0 is the default.
* Sky shell: +1.4 dB on the yard.
* Colour correction off: raw held-out PSNR +1.2 dB but colour-fit PSNR −0.5 dB (settled below).
* `inFlight: 2`: same quality, a few it/s faster on desktop; the default stays 1 (iOS).

GPU time per full-resolution iteration (600×800, 300 k Gaussians): project 0.5 ms, depth sort 1.0, binning 0.6, tile
sort 0.7, rasteriser 3.7, loss 1.6, rasteriser backward 10–11, projection backward + Adam 1.5: ≈ 20 ms, 39–48 it/s
(target was 15–30 it/s at 0.5–0.7 MP). Rasteriser backward workgroup size: 256 threads 12.0 ms, 128 10.9, 64 10.0
(default), 32 13.8; two splats per thread step 15.7. Float CAS adds cost ≈ 1 ms; exact tile culling and the σ
early-out were within noise. Yard iterations are ~2× cheaper (0.29 M vs 0.64 M tile intersections).

## Engine splat stage

`run-e2e.mjs object-real --engine worker --splat --nvs` (fast preset; e2e/README.md "Splat stage in the engine"):
stage 60.3 s = prepare 12.6 s (initial Gaussians 1.1 s: 90 113 dense + 12 316 sparse; 90 view uploads 11.5 s,
≈ 128 ms per view on the engine thread) + training 47.5 s (3 000 iterations, 64.6 it/s over the coarse-to-fine
schedule, 53.8 it/s at full 600×800, 6 checkpoints ≈ 1.0 s) + finishing 0.1 s; 200 000 Gaussians. Memory plan
peak ≈ 811 MB of 1.2 GB (training 336 MB: GPU 301 MB — images 165, pixels 40, intersections 37, Adam 24 — + host
35 MB). Checkpoints: 112 B per Gaussian, 22 MB at 200 k, 168 ms each (Chromium, scratch IndexedDB).

Novel views, same run, 24 held-out frames at 600×800:

| renderer | coverage | full PSNR / SSIM | covered PSNR / SSIM | cc full PSNR | cc covered PSNR | shift px |
|---|---|---|---|---|---|---|
| `points` | 0.763 | 10.39 / 0.475 | 24.02 / 0.755 | 10.41 | 24.52 | 0.19 |
| `mesh-lit` | 0.610 | 8.65 / 0.530 | 23.55 / 0.877 | 8.68 | 25.50 | 0.13 |
| `mesh-unlit` (now the app default) | 0.610 | 8.71 / 0.552 | 28.65 / 0.915 | 8.72 | 30.00 | 0.08 |
| `splat` | 0.999 | 31.24 / 0.958 | 30.62 / 0.950 | 33.74 | 33.44 | 0.08 |

Splat full-frame PSNR per view: median 31.13, min 23.36, max 35.14 dB. The splats fill the frame and beat the unlit
mesh by 2 dB on its own pixels. Held-out frames sit 1–3 frames from a training keyframe, which favours image-fitted
representations; the off-trajectory sets below measure extrapolation.

### Memory plan examples

`planSplat` / `fitSplatPlan` (src/engine/memory.ts), 90 keyframes / 160 / 300 available, 40 MB resident:

* fast: fits 1.2 GB and 850 MB unchanged (GPU 301 MB, peak 626 MB); 600 MB → 64–73 views.
* balanced: 1.2 GB keeps 160 views at 1024 px (GPU 689 MB, peak 1032 MB); 850 MB → 97 views.
* max: 1.2 GB → ≈ 100 views at 1280 px, 500 k Gaussians (peak ≈ 1.19 GB); 850 MB → 45–100 views, 1037–1280 px,
  250–361 k Gaussians.

Geometry supervision adds the depth targets (40 × 300 × 400 × 8 B = 37 MB at fast), 32 B per training pixel and 4 B
per Gaussian: fast object GPU 301 → 353 MB, peak 686 → 738 MB.

## Before / after and off-trajectory views

e2e/README.md "Breakthrough results" (fast, two runs each): post-capture processing 202.9 → 156.1 s on object-real and
149.8 → 112.2 s on yard-real with the splat stage (96.5 / 70.4 s without it). The splat view scores 31.3 / 31.1 dB
full frame on held-out frames against 8.6 / 9.8 dB for the lit mesh. Off trajectory (same seeded scene, shifted
paths; `nvs.mjs --test-dataset`) it changes by +0.1 to −3.3 dB (object ±5 cm radius, +12 cm height) and −1.3 to
−8.3 dB (yard ±1.5 m sideways, +0.75 m height); the stress cases lose 8.0 dB (object +25 cm) and 11.0 dB (yard
+1.5 m). Height is the weak axis: the captures never see the object top, tree crowns or the ground behind the
fence. Even so the splats stay 9.8–21.2 dB above `mesh-lit`, and ahead of the unlit mesh on the mesh's own pixels
except at object +25 cm (−0.4 dB).

## Geometry supervision, shape regularisers, colour correction

e2e/README.md "Geometry supervision, shape regularisers, colour correction" (25 variants via `e2e/splat-replay.mjs`
on one frozen session per scene). Rule: a change is on by default only if it raises the mean off-trajectory
full-frame PSNR, lowers held-out by ≤ 0.2 dB and costs ≤ ~15 % splat-stage time. Full-frame PSNR; off = mean of the
four off-trajectory sets; drift-free = the capture path and the +12 cm / +0.75 m path without exposure /
white-balance drift; Δ against the five "before" runs.

| variant | object held / off (Δ) | yard held / off (Δ) | drift-free object path / +12 cm, yard path / +0.75 m | stage time |
|---|---|---|---|---|
| before (colour correction on, no geometry) | 31.28 / 27.10 | 31.10 / 22.83 | 32.53 / 27.81, 30.27 / 20.62 | 59.4 / 40.2 s |
| depth + normal, whole run | 31.30 / 26.99 (−0.11) | 30.84 / 22.65 (−0.18) | | +24 % |
| depth only, until 0.5 | 31.31 / 27.09 (−0.01) | 31.04 / 23.21 (+0.37) | | |
| depth + normal, until 0.3 | 31.27 / 27.10 (0.00) | 30.99 / 23.31 (+0.48) | 32.56 / 27.87, 30.26 / 21.32 | +4 / +7 % |
| **depth only, until 0.3 (default)** | **31.37 / 27.25 (+0.15)** | **31.01 / 23.31 (+0.48)** | **32.62 / 28.09, 30.25 / 21.12** | **+4.1 / +6.4 %** |
| depth only, until 0.3, grazing cos² | 31.32 / 26.95 (−0.15) | 31.03 / 23.32 (+0.49) | | |
| flatten 100 | 31.35 / 27.05 (−0.05) | 31.13 / 22.84 (+0.01) | | |
| anisotropy cap 5 | 31.29 / 27.12 (+0.01) | 31.02 / 22.80 (−0.04) | | |
| colour correction off | 33.76 / 27.97 (+0.87) | 31.92 / 22.78 (−0.06) | 30.72 / 27.40, 29.80 / 20.25 | |
| colour correction off + depth only, until 0.3 | 33.41 / 27.82 (+0.72) | 31.78 / 23.34 (+0.51) | 31.02 / 27.34, 29.87 / 20.99 | |

Decisions:

* **Inverse-depth supervision during the first 30 % is on by default** (mean off-trajectory +0.31 dB, held-out
  +0.09 / −0.09 dB, stage +4–6 %, GPU +52 MB at fast). Supervising the whole run punches holes into ground the capture
  only saw at grazing angles (yard inward −1.7 dB, coverage 0.981); stopping early avoids it. The normal term, the
  grazing weights, `flatten` and `anisotropy` stay options (no gain). On the un-normalised blended inverse depth (no
  division by the accumulated alpha) the loss can only be met where the render is opaque (toy test: depth error
  −25 % vs −94 % with the normalisation).
* **Shape regularisers act but do not help**: object Gaussians with mid/min > 5 73.5 → 99.7 % with flatten 100;
  max/mid > 5 8.3 → 0.1 % with the cap 5; every PSNR within noise.
* **Per-view colour correction stays on**: it loses raw held-out and regular off-trajectory PSNR only because those
  test frames reproduce the capture's exposure drift on the same clock, which an uncorrected model bakes into
  region-wise colours; on drift-free views the correction wins by 0.4–1.8 dB. Use `--exposure-jitter 0 --wb-drift 0`
  test sets for appearance questions.
* **What remains** is unseen content (object top / floor at +25 cm, tree crowns and the ground behind the fence from
  +1.5 m): depth supervision recovers 0.5–0.9 dB of the yard's 8.3–11 dB height gap; a generative or symmetry prior
  would be needed for the rest.

## Viewer splat layer

`e2e/ui/splat-shots.mjs` (e2e/README.md "Viewer splat layer"), iPhone emulation 393×852 @3x → splat canvas
786×1704 = 1.34 MP, 200 k Gaussians: submit → done per full-size frame while orbiting, median / p90 16.3 / 22.5 ms on
object-real (0.93 M intersections), 12.7 / 18.1 ms on yard-real (0.34 M); layer GPU memory 107 MB (reservation 3.2 M
intersections = 51 MB, pixels 27 MB). On a phone a full frame is expected to take several times longer — the reason
for the 0.5 MP interactive frames.

## Open questions

* iPhone: iteration time (split forward / backward if one command buffer exceeds ~30 ms), IndexedDB checkpoint writes
  of 22–56 MB every 500 iterations, budget headroom of `max`, viewer frame time.
* View preparation on the engine thread (≈ 128 ms per 600×800 view, 19 % of the fast stage): CPU helper workers or
  reuse of the dense stage's undistorted views.
* Depth-sorting only the visible Gaussians, 16-bit depth keys, f16 Adam moments.
* Measuring on splats: the geometry forward renders D / A per pixel, so a tap could pick the expected inverse depth of
  the splats (1-pixel readback) instead of the hidden dense points / mesh.
