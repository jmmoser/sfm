# Feature extraction and matching: measurements (CPU and WebGPU)

Measurements from 2026-10; the code may have changed since — check the source before relying on a number.

Code: `src/features/` (sift-cpu.ts, simd.ts, match.ts, sift-gpu.ts, match-gpu.ts, extractor.ts).

## CPU reference (Node 22, x86, single thread)

| Operation | WASM SIMD | plain JS |
| --- | --- | --- |
| SIFT extraction, 1600×1200 → 4096 features | ≈ 420 ms | ≈ 1.4 s |
| Brute-force matching, 4000 × 4000 | ≈ 180 ms | ≈ 2.4 s |
| Guided matching, `positions` (r = 20 px) | 5–15 ms | |
| Guided matching, `F` (3 px) | 30–45 ms | |

Descriptor norm recomputation per match call: O(128 n), < 1 ms for 4k features.

CPU scratch arena (`releaseSiftCPUMemory`): ≈ 110 MB for a 1600×1200 image (or a < 800 px image
upsampled to octave −1).

## CPU accuracy (unit tests)

* Gaussian-blob localisation bias < 0.03 px in the full-resolution frame for all processing paths
  (internal / external downscale, octave −1 / 0).
* Precision ≥ 0.957 and repeatability 0.55–0.99 under rotation 30° / 90°, scale 0.6 / 1.5,
  tilt 40°, photometric change and noise (procedural textures and warps from `testdata.ts`).
* Pinned conventions (not caught by matching precision): orientation sign, exact-90° equivariance,
  absolute descriptor layout incl. y′ sign and orientation-bin direction (mutation-tested:
  transposed bins, flipped x′ / y′ and flipped (θ − φ) all fail), S = 2 / 4 localisation and DoG
  scale 2^(−1/(2S)), F-guided band walk with a finite epipole (steep and flat lines) against a
  brute-force reference.

## GPU vs CPU parity (Chromium tests)

SIFT, 640×480 textures:

* identical feature counts, 100 % of CPU keypoints reproduced (max position difference 4e-3 px);
* median descriptor cosine 1.00000, ≈ 70 % of descriptors byte-identical, the rest differ by ±1 in a
  few bytes (f32 vs f64);
* warped-pair matching precision identical to the CPU pipeline.

Same result (identical counts, position agreement 1.000, median cosine 1.00000) for S = 1 / 2 / 4,
`maxOrientations` 4, σ below the base blur (copy path), `firstOctave` 1, `distribute: false`, odd
sizes and a 700×40 image. Tiled and direct blurs are bit-identical at odd sizes and with radius 45
(above the tiled vertical-blur limit). A submit budget giving 63 command buffers is bit-identical.
Test thresholds: position ≥ 0.97, cosine ≥ 0.995.

Matcher: 100.00 % overlap with matchCPU / matchGuidedCPU in all tests, incl. real 4096 × 3366 SIFT
sets, multi-pass matching (`rowTile0 > 0`), extreme column chunking, exact ties, all-zero
descriptors and `ratio > 1`. Guided tile culling evaluates 2–4× fewer pairs on warped pairs.

## GPU memory (estimates)

* SiftGPU, per cached processed size (`maxCachedSizes` 2): (S+3) planes × Σ octave pixels × 4 B +
  one temp plane. 1600×1200: ≈ 69 MB; 640×480 (upsampled to 1280×960): ≈ 44 MB.
* Plus the input: u8 1 B/px, f32 4 B/px (f32 input already at the processed size is written straight
  into the pyramid), bitmap RGBA8 texture 4 B/px.
* Plus ≈ 3 MB fixed: candidates 65 536 × 32 B, keypoint / orientation / descriptor buffers for
  maxFeatures × maxOrientations.
* MatcherGPU: descriptor cache budget (default 64 MiB) + 16 B × (na + nb) × column chunks per call.

## SwiftShader timings (CPU emulation, informational only)

| Operation | Time |
| --- | --- |
| SIFT 640×480 (octave −1) | ≈ 2.4–4.4 s |
| SIFT 320×240 | ≈ 0.6 s |
| Matching 2000 × 2000 | ≈ 2.2 s |
| Matching 4096 × 3366 | ≈ 3.4 s |

Shared-memory tiled blurs are ≈ 10× slower than direct loads on SwiftShader (barrier emulation),
hence `cfg.blur: 'auto'` picks direct on software adapters.

No on-device (A-series) profiling yet: tile sizes, `submitBudget` and the tiled-vs-direct blur
choice should be tuned on an iPhone (`cfg.profile` records per-kernel times).
