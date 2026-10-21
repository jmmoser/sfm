# On-device benchmark, memory budget and headless engine timings

Measurements from 2026-10; the code may have changed since — check the source before relying on a number.

Nothing here was measured on an iPhone. Headless numbers come from Chromium with SwiftShader (software
WebGPU) and say nothing about Apple GPUs; the memory figures are analytic estimates
(`src/engine/memory.ts`) calibrated on desktop runs.

## Diagnostics benchmark (`src/app/bench/`)

What it runs (sizes per scale in `src/app/bench/scales.ts`: `full`, `small` = `?benchScale=small` / CI,
`tiny` = Node tests):

1. **SIFT**: WebGPU extraction of a procedural 1920×1440 image (`upsampleBelow: 0`, so cost ∝ pixels;
   warm-up + median of 3; stage timings upload / detect / select / describe / assemble). GPU-vs-CPU parity
   at 640×480: feature count within ±5 %, ≥ 90 % of keypoints within 1 px, median descriptor cosine ≥ 0.97.
   Also the CPU speed.
2. **Matcher**: GPU 4096 × 4096 synthetic RootSIFT descriptors (60 % noisy copies + distractors; first call
   incl. upload, cached call) vs `matchCPU`: ≥ 99 % identical pairs, recall ≥ 80 %, precision ≥ 98 %.
3. **PatchMatch**: `makeSyntheticDataset('near')` 640×480, 8 sources, 4 iterations; per-iteration time,
   accuracy vs ground truth on `surfaceMask` (valid ≥ 70 %, within 1 % ≥ 80 %); GPU-vs-CPU agreement at
   160×120 (≥ 70 % within 2 %); CPU PatchMatch speed.
4. **BA**: `makeSyntheticBA` 100 images / 20 k points (strafe), 8 LM iterations; time per iteration, final
   RMS < 0.8 px.
5. **Scan-time estimates** per preset (`estimate.ts`): a rough model over `QUALITY_PRESETS` /
   `resolveDensePreset`; the coefficients are assumptions, unverified on devices.

### Headless results (SwiftShader, `small`, 4 cores)

| test | GPU | CPU |
|---|---|---|
| SIFT | 480×360 ≈ 0.95 s | 320×240 ≈ 30 ms |
| matcher | 1024² ≈ 1 s | ≈ 30 ms |
| PatchMatch 160×120, 4 sources | ≈ 0.5–0.65 s per iteration | 2 iterations 1.5 s |
| BA, 30 images | — | ≈ 20–30 ms per iteration |

Parity: 100 % keypoints / descriptor cosine 1.000, 99.1 % depth agreement, 100 % identical matcher pairs.
The UI stayed responsive during the run (61 rAF/s; the benchmark runs in its own worker).

## Engine on SwiftShader (e2e harness, headless Chromium)

* GPU SIFT ≈ 4–8 s per 720×960 frame and GPU matching ≈ 2 s per pair, vs CPU ≈ 0.3–0.5 s / ≈ 0.05 s —
  why the e2e runs use `EngineConfig.featureBackend: 'cpu'`. The GPU feature path through the worker
  (e2e `object`, every 2nd frame) gives the same accuracy class (ATE 2.0e-4 of the extent, rotation 0.039°,
  scale +0.86 %) at ≈ 25 s per frame.
* Dense: ≈ 25–45 s per 480 px PatchMatch view.
* `maxMappingBacklog` (default 2): without back-pressure, background mapping fell behind the keyframe rate
  on the e2e captures — only 31/90 (object) and 36/80 (yard) keyframes were mapped live and
  `finishCapture` took 20–28 s to drain. With it: 90/90 and 80/80 live, `finishCapture` < 1 s; the frame
  rate drops instead.
* Principal point refined in the final BA of refine (true offset 1.8 px on the e2e datasets): mean
  rotation error 0.135° → 0.054° (object) and 0.149° → 0.059° (yard), principal-point error
  1.80 → 0.3–0.5 px, yard ATE 1.26 → 0.88 cm.

Accuracy results of the full e2e runs: `e2e/README.md` ("Engine end-to-end results").

## Memory budget (estimates, `src/engine/memory.ts`)

Measured on e2e `object` (`balanced`, 720×960, ≈ 1.6 k features / keyframe): 0.55 MB / keyframe after
capture (features 233 KB, graph 178 KB, reconstruction 100 KB, pairs 33 KB), 0.6 MB after refine.

iPhone estimates scale this to 4 096 features / keyframe at 1920×1440 `balanced` ≈ 1.4 MB / keyframe.
Budget 1.2 GB tab with a 250 MB page / JS reserve:

| stage | resident / peak | main items |
|---|---|---|
| capture, 300 keyframes | ≈ 430 MB state + ≈ 145 MB GPU + ≤ 60 MB per-frame transients + reserve ≈ 885 MB | features 180, graph 135, live model 75, pairs 26; GPU: SIFT pyramids ≈ 70 + input 11 + matcher cache ≤ 64 (CPU tier: SIFT arena 110) |
| refine | ≈ 1.0 GB | state + refine pairs ≈ 45 + 2nd reconstruction 75 + working copy 75 + global BA ≈ 85 + matcher cache 64 |
| dense `max`, 150 views 1600×1200, PatchMatch | 663 MB + reserve | image LRU 176, depth-only sources 120, prior + output + serialised + halo copies 132, colour thumbnails 96, PatchMatchGPU 140 |
| dense `max`, fusion | 822 MB + reserve (step 5) | 10 resident maps 366 + store 38, masks 34, colour 96, filtered output 59, ≈ 2.3 M points × 105 B |
| mesh (`max`) | 570 MB + reserve | TSDF 40 k blocks 215, scratch 29, maps 110, colour 96, marching cubes / simplification ≈ 120 |
| dense `balanced`, 80 views 1280×960 | fusion 895 MB + reserve (step 2) | 10 maps 258, ≈ 4.7 M points 492 |

Capture cap (`captureMemoryFraction` 0.45 of budget − 250 − 64 MB): ≈ 400 MB at 1.2 GB (≈ 280 keyframes
at `balanced`, ≈ 150 at `max`), ≈ 240 MB at 850 MB; refine peaks at ≈ 1.9× the capture state.

Dense check after the memory caps (e2e `object` through the worker, 480 px, CPU features, IndexedDB depth
store): 1 307 864 fused points (plan expected 1.38 M, step 1), P / C / F 0.9996 / 0.9966 / 0.9981, mesh
488 k triangles, ATE 2.06e-4 — unchanged by the caps.

## Not yet verified on a device

* Memory: no on-device profile (no `performance.memory` on iOS); `maxBufferSize` budget classes.
* Timings: PatchMatch per 1600 px view (estimated 4–15 s), scan-time estimate coefficients, splat-layer
  frame time on an iPhone.
* App: wake lock from a tap that navigates to Processing; `navigator.share` of 100+ MB exports; iPad
  landscape layout of the capture controls; Safari's `inert` / focus behaviour with VoiceOver; whether a
  `devicemotion` / camera permission prompt makes the page `hidden` (it should not).
