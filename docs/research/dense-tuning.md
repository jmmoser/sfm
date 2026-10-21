# Dense reconstruction: tuning and measurements

Measurements from 2026-10 (synthetic unit-test scenes, the GT-pose dense harness, and an M4 Metal GPU); the code may have changed since — check the source before relying on a number.

The defaults and rationale live in the source (`src/dense/presets.ts`, `src/dense/depth-filter.ts`,
`src/dense/pointcloud.ts`, …). This file holds the evidence behind them.

Contents:

1. [Unit-test measurements (synthetic scenes)](#unit-test-measurements-synthetic-scenes)
2. [Dense tuning results](#dense-tuning-results)
3. [Sky / low-texture halo removal](#sky--low-texture-halo-removal)
4. [Dense on a real GPU](#dense-on-a-real-gpu)

## Unit-test measurements (synthetic scenes)

Node 22, x86, one core unless stated; tests in `src/dense/*.test.ts`.

### PatchMatch

"Surface" = GT pixels > 5 px from depth edges with a viewing angle < 75°; 1 reference + 6 sources.

* CPU 160×120 near scene: surface 99.8 % valid, 89 % within 1 %, median normal error 5°; all pixels
  71 % within 1 % (window-radius fattening / rim effects dominate at this resolution); 5.6 s.
* GPU 256×192 near: surface 94 % within 1 %, all pixels 85 %, normals 4.3°. Far scene (GT 6.5–56 m,
  search 4–76 m): surface 90 % within 1 %, 97 % within 2 %.
* Geometric pass (192×144): all-pixel accuracy 78 → 82 % within 1 %, surface 92 → 95.5 %.
* 640×480 / 8 sources: 97 % within 1 % after 2 iterations.
* GPU vs CPU: ≈ 70 % of pixels within 1e-4, all metrics within 0.2 %.
* SwiftShader runs ≈ 1 s per iteration at 256×192 (CPU emulation, not representative of Apple GPUs).
* Multi-scale recipe on the near scene: feeding ½-resolution maps as `sourceDepths` of a
  full-resolution geometric pass gives 62 % within 1 % (plain full-resolution photometric: 88 %).
  A prior-initialised photometric pass (2 iterations from the ½-res map) matches 6 random-init
  iterations (88.2 % vs 87.5 %) at about 40 % of their cost.
* Source scale mismatch (no mip selection): a source at 2× the reference resolution drops
  within 1 % from 81 % to 70 %; at 0.5×, to 33 %.
* `affineWindow` with the 13 px window at 160×120 (window spans 8 % of the image): −5 pt within 1 %
  (the unit test uses the 11 px window for the affine check).

### Fusion

* Plane-depth test vs raw neighbour depth: 87.0 → 85.4 % within 0.5 % without the plane test.
* GT depth maps (σ 0.2 %, 3° normal noise, 10 % outliers), 16 views 160×120: 99.99 % of points
  within 0.5 % of the true surface, p99 0.26 %, 0 points > 2 %; filtered depth maps 0.001 % bad pixels.
* 640×480 × 16 views, 6 neighbours: 1.6–1.9 s (0.33–0.38 µs/px), 746 k points, 100 % within 0.5 %.
* Real `patchMatchCPU` output (photometric + geometric, near scene 256×192, 10 views, 4 sources):
  fused 99.53 % within 1 %, 96.9 % within 0.5 % (128×96: 94.9 / 87.0 %, window fattening dominates).

### Undistortion

Barrel round trip 0.31 grey levels mean |Δ| vs a direct pinhole render; dot centroid error
0.014 / 0.031 / 0.022 px at full / ½ / ⅓ size; 320×240 in 14 ms.

### TSDF and marching cubes

* 32 GT depth maps of a sphere: closed genus-0 mesh, radius error mean 0.015, max 0.095 voxel
  (0.024 / 0.14 before the grazing-surface reach). σ 0.3 % noisy maps: mean 0.041 voxel. Yawed box
  p95 0.54 voxel (corners rounded by < 1 voxel; box edges seen at grazing angles get slightly
  rounder with the extended reach, 0.48 without). Budget of 120 blocks → 2 coarsening levels, still
  closed. A ×50 scaled sphere gives identical relative error.
* Near scene 400×300 × 16 views, 7.5 k blocks: integration ≈ 1.3 s (≈ 30 ns per voxel visit),
  marching cubes ≈ 0.6 s for 708 k triangles.
* Allocation per-pixel sampling, near box + wall 15× further: 0 missing blocks, far wall 100 % meshed
  (a single stride from the 2nd depth percentile missed 1 223 of 1 319 wall pixels and 285 box pixels,
  57 % coverage).
* Grazing ground (camera 1.6 m above a plane), with the extended reach behind the surface: yard
  largest component 50 k → 157 k triangles, all components 171 k → 293 k; on a pure ground plane the
  share of ground within 25 m covered by the largest component 73 % → 96.5 %.
* Tile culling with f·r·√(1 + (X/Z)²)/(Z − r): voxel update count matches a brute-force integrator
  exactly (ray and plane mode).
* Marching cubes on analytic SDFs: sphere χ = 2, torus χ = 0, max vertex error 0.006 voxel, 0 boundary /
  non-manifold / misoriented edges, also on a high-frequency noisy SDF with 419 components (plain fan
  triangulation had 14 non-manifold edges there).
* Small components: a 7 m tree at 40 m is 3.1 k triangles = 1.99 % of the largest component, hence
  `componentKeepVoxels`.

### Simplification and point-cloud filters

* Simplification to 25 %: two-sided Hausdorff ≤ 0.033 voxel on a 23 k-triangle sphere, colours within
  1.1 levels, no flips, closed; 285 k triangles in ≈ 1.3–1.4 s (≈ 4.7 µs/input triangle, memory-bound).
* SOR k = 8: 1 M points in 2.3 s (k-d tree build 0.5 s); 93 % of injected outliers removed, 0 inliers.
  Multi-scale test: the plain statistic removed 196/2500 far-wall points and 0/100 outliers; the
  scale-adaptive one 0 and 100/100.

## Dense tuning results

### Setup

`e2e/run-dense-gt.mjs` with exact poses, so every error is MVS or fusion. Reference ("full") runs:
480 px; object 24 views (360×480), yard 30 views (480×360), stride 8; 8 sources; 6 photometric +
3 geometric iterations. τ = `denseTau` (object 13.6 mm, yard 0.1 m); P / C / F = precision /
completeness / F-score at τ, identity alignment. Fusion experiments re-fuse the saved reference maps
(`--load-maps`, seconds each). PatchMatch experiments run on "mini" subsets at 480 px (yard frames
0–112 = 15 views, object frames 0–88 = 12 views). Each is re-fused with the reference and with the
tuned fusion, and compared on Csub, the completeness of the GT samples visible in the used views.
SwiftShader was shared with other jobs (load 5–12), so timings are relative only. One change at a
time; a change was kept only when it helped object AND yard without costing the other metric
materially. Harness options: `--preset`, `--geom-passes`, `preHalo`, `filter`, halo-rejection
statistics per GT error class.

### Fusion experiments (full maps; P / C / F)

| # | Setting | object | yard |
|---|---|---|---|
| 0 | reference: `minViews` 3, normal 10° | .9991 / .9601 / .9792 | .9885 / .2307 / .3741 |
| 1 | `minViews` 2 | .9982 / .9875 / .9928 | .9777 / .3376 / .5019 |
| 2 | 1 + normal 20° · 30° · off | .9978 / .9889 / .9934 · .9976 / .9898 / .9937 · .9965 / .9924 / .9944 | .9641 / .3599 / .5242 · .9541 / .3691 / .5323 · .9122 / .3846 / .5411 |
| 3 | 1 + low-angle rule (largest triangulation angle < 3° · 5° ⇒ 3 views) | .9982 / .9875 / .9928 · .9983 / .9875 / .9929 | .9846 / .3169 / .4795 · .9879 / .2964 / .4560 |
| 4 | 1 + halo filter, colour-mean rule, r 4 (dropped) | .9986 / .9867 / .9926 | .9802 / .3349 / .4993 |
| 5 | 1 + halo filter, vote rule, r 4 · r 8 step 2 | .9985 / .9865 / .9925 · .9996 / .9872 / .9933 | .9809 / .3355 / .5000 · .9828 / .3343 / .4989 |
| 6 | 5 (r 8) + normal 12° · **15°** · 20° | .9997 / .9877 / .9936 · .9995 / .9882 / .9939 · .9995 / .9885 / .9939 | .9814 / .3402 / .5053 · .9780 / .3476 / .5129 · .9743 / .3556 / .5211 |
| 7 | 1 + normal 15°, no halo filter | .9980 / .9886 / .9933 | .9705 / .3515 / .5160 |
| 8 | **6 (15°) with halo step 4** (3× cheaper, same result) = tuned fusion | .9996 / .9883 / .9939 | .9788 / .3473 / .5127 |
| 9 | 8 + `maxDepthError` 0.02 | .9996 / .9887 / .9941 | .9771 / .3520 / .5175 |
| 10 | 8 + `maxNeighbours` 6 · 8 (reference 10) | .9997 / .9870 / .9933 · .9997 / .9880 / .9938 | .9794 / .3473 / .5127 · .9788 / .3475 / .5129 |
| 11 | 8 + SOR k 8, σ 1.5 · 2.5 · 4 | .9998 / .9844 / .9920 · .9998 / .9865 / .9931 · .9997 / .9877 / .9937 | .9838 / .3282 / .4922 · .9816 / .3386 / .5035 · .9801 / .3438 / .5090 |
| 12 | 8, halo filter applied twice (map + fusion) | .9998 / .9880 / .9939 | .9819 / .3440 / .5095 |
| 13 | 8 on photometric maps (no geometric pass) | .9997 / .9880 / .9938 | .9821 / .3351 / .4997 |
| 14 | 8 on yard maps computed with `depthMarginMin` 0.5 | — | .9809 / .3522 / .5183 |

TSDF mesh of the fusion-filtered maps, P / C / F: object .990 / .938 / .963 → .999 / .974 / .987
(tuned); yard .898 / .216 / .348 → .844 / .340 / .484. Building the TSDF input with `minViews` 3
(cloud with 2) is worse on both: object .9998 / .943 / .970, yard .883 / .245 / .384.

### PatchMatch experiments

Mini subsets; Csub / P with the reference fusion → with the tuned fusion; "≤1 % / >5 %" =
geometric-pass pixels within 1 % / off by more than 5 % of the surfel GT; PM s = PatchMatch seconds,
contended SwiftShader.

| run | yard Csub / P: ref fusion → tuned | yard ≤1 % / >5 % | object Csub / P: ref fusion → tuned | object ≤1 % / >5 % | PM s yard / object |
|---|---|---|---|---|---|
| reference settings (8 sources, 6 + 3 iterations, margin 0.8) | 0.3083 / 0.9905 → 0.5234 / 0.9797 | 84.2 / 9.8 | 0.8296 / 0.9999 → 0.9166 / 0.9997 | 84.4 / 8.7 | 726 / 969 |
| `depthMarginMin` 0.5 (base of the rows below) | 0.3076 / 0.9906 → 0.5277 / 0.9810 | 87.7 / 6.5 | 0.8235 / 1.0000 → 0.9167 / 0.9998 | 84.0 / 9.3 | 914 / 1214 |
| multi-scale (½-res 6 it → full-res 2 it prior → geometric) | 0.3102 / 0.9899 → 0.5286 / 0.9795 | 88.2 / 6.2 | 0.8209 / 0.9999 → 0.9151 / 0.9996 | 84.8 / 8.4 | 823 / 643 |
| 5 sources | 0.3041 / 0.9903 → 0.5253 / 0.9810 | 87.6 / 6.5 | 0.8176 / 1.0000 → 0.9085 / 0.9995 | 83.0 / 9.9 | 686 / 702 |
| 4 photometric iterations | 0.3065 / 0.9909 → 0.5242 / 0.9811 | 87.8 / 6.4 | 0.8223 / 1.0000 → 0.9151 / 0.9998 | 83.9 / 9.5 | 767 / 754 |
| 2 geometric iterations | 0.3087 / 0.9906 → 0.5287 / 0.9809 | 87.8 / 6.4 | 0.8223 / 1.0000 → 0.9155 / 0.9998 | 84.1 / 9.3 | 970 / 1178 |
| 2 geometric passes | 0.3139 / 0.9897 → 0.5299 / 0.9796 | 88.0 / 6.5 | 0.8323 / 1.0000 → 0.9157 / 0.9997 | 84.7 / 9.2 | 1066 / 1160 |

### Kept

1. `minViews` 2 for object and scene mode: the largest gain. Keyframe-spaced views rarely give
   3 agreeing maps; with GT depth fed into fusion, C goes 0.316 → 0.461 on yard and 0.983 → 0.998
   on object.
2. The colour-guided halo filter (`depthHaloMask` / `removeDepthHalos`) on the final maps. It rejects
   66–68 % of the > 5 % outliers and 81 % of the sky estimates, for 0.9–1.2 % of the ≤ 1 % pixels.
   With `minViews` 2 and 10°, yard points farther than τ fall from 21.4 k to 16.3 k, and 0–10 m
   precision is 99.8–100 %. Object points farther than τ fall from 388 to 192, which removes most of
   the visible beige silhouette halos. Cost ≈ 45 ms per 480 px map.
3. Normal threshold 15° with the halo filter. PatchMatch normals of far and grazing surfaces are
   noisy, and 15° buys back the completeness the filter costs (row 8 vs row 1: object +0.14 P /
   +0.08 C, yard +0.11 P / +0.97 C, in points).
4. `depthMarginMin` 0.5 in view selection. The 1st-percentile sparse depth misses the near ground in
   25 of 30 yard views: the GT sparse samples are area-uniform, and real SfM tracks are short there.
   With the wider bound, the yard maps lose a third of their > 5 % outliers (11.5 → 7.7 %) and 0–5 m
   completeness rises from 56 to 68 %. On object it is neutral with the tuned fusion (Csub 0.9166 →
   0.9167), but −0.6 pt with the reference fusion.
5. Cheaper PatchMatch for `fast`, measured with the tuned fusion (object / yard Csub):
   4 photometric iterations −0.16 / −0.35 pt; 5 sources −0.8 / −0.24 pt and −25–38 % PatchMatch time;
   2 geometric iterations −0.12 / +0.10 pt.

### Not kept

* Low-angle rule: trades far-field completeness for precision at about 3:1.
* SOR: removes sparse but correct far points (yard −0.9 pt C for +0.3 pt P at σ 2.5), so the presets
  set `sorK` 0.
* `maxDepthError` 0.02: yard +0.5 C / −0.2 P, object ±0, too small to decide on.
* Applying the halo filter twice: erodes edges (yard −0.3 pt C).
* Multi-scale, on SwiftShader: quality-neutral (yard Csub +0.1 / P −0.15, object −0.16 / −0.02), 25–45 %
  cheaper in the photometric stage. Later enabled in every preset once the engine got a multi-scale
  driver (see [Dense on a real GPU](#dense-on-a-real-gpu)).
* A second geometric pass: +0.6 / +0.9 pt Csub with the reference fusion, but within ±0.2 pt with the
  tuned one (yard +0.2 / P −0.14, object −0.1), for +33 % PatchMatch time. `max` keeps one pass.

The function defaults (`DEFAULT_FUSION_OPTIONS`, `DEFAULT_VIEW_SELECTION_OPTIONS`,
`DEFAULT_PATCHMATCH_OPTIONS`) were left unchanged because existing tests pin them (`minViews` 2 or 15°
breaks `fusion.test` / `dense-e2e.test`); the tuned values live in `DENSE_PRESETS`.

### Final vs reference (full views, 480 px, `--preset balanced`)

| | object P / C / F, acc med / p90, mesh F | yard P / C / F, acc med / p90, mesh F |
|---|---|---|
| reference | 0.9991 / 0.960 / 0.979, 0.11 / 0.59 mm, 0.963 | 0.9885 / 0.231 / 0.374, 3.4 / 31 mm, 0.348 |
| tuned | 0.9997 / 0.987 / 0.994, 0.13 / 0.63 mm, 0.987 | 0.9809 / 0.352 / 0.518, 2.9 / 33 mm, 0.492 |

Yard by distance to the nearest camera, C (P): 0–5 m 31 → 68 % (100 → 100 %), 5–10 m 52 → 76 %
(99.9 → 99.8 %), 10–20 m 42 → 61 % (99.4 → 97.9 %), 20–40 m 24 → 36 % (94 → 88 %), > 40 m 3.6 → 6.2 %
(65 → 53 %).

The precision cost of `minViews` 2 is far-field depth noise at 480 px: the wrong far points are
1.6–1.8 τ off (median), not gross errors. A sky-coloured fringe also remained on roof and tree
outlines (28 % of the remaining > τ points), addressed by the sky test below.

The object's accuracy median and p90 rise slightly (0.11 → 0.13 mm, 0.59 → 0.63 mm) because 2-view
points are noisier. Precision at τ rises, and the focus box stays at 99.96 %.

### Preset cost model

| preset | dense max side | max views | sources | photometric it | geometric passes × it | fusion neighbours | work per view (MP·src·it) | SwiftShader est. per view | iPhone est. (30–100×) |
|---|---|---|---|---|---|---|---|---|---|
| `fast` | 800 | 40 | 5 | 4 | 1 × 2 | 6 | 14 | ≈ 45 s | 0.5–1.5 s per view, ≤ 1 min for 40 views |
| `balanced` | 1280 | 80 | 8 | 6 | 1 × 3 | 8 | 89 | ≈ 4.7 min | 3–9 s per view, 4–13 min for 80 views |
| `max` | 1600 | 150 | 8 | 6 | 1 × 3 | 8 | 138 | ≈ 7.4 min | 4.5–15 s per view, 11–37 min for 150 views |

(Work figures are for single-scale runs; multi-scale replaces most full-resolution photometric
iterations with ½-resolution ones.) Image sizes and view counts were kept from the engine's earlier
values because resolution and views are the levers for far content; on SwiftShader they could only be
checked at 480 px.

PatchMatch time ∝ MP × sources × (photometric + geometric iterations); one geometric iteration costs
about the same as a photometric one. On SwiftShader (4 cores, uncontended) ≈ 3.2 s per
MP·source·iteration. The iPhone factor (30–100× faster) is an assumption, not a measurement. Fusion
plus the halo filter take ≈ 0.6 µs per pixel per view on one x86 core. Fusion keeps
(`fusionMaxNeighbours` + 2) × 20 B/px resident, ≈ 370 MB at 1600 px with 8 neighbours — hence 8, not
10, neighbours for `max` (row 10: no quality loss).

### Far vs near range

Nothing in the pipeline assumes a depth scale: depth ranges are per view with inverse-depth sampling;
the fusion tests are relative (1 % depth, 2 px, 15°); the halo filter works in pixels and relative
depth; the TSDF voxel is automatic.

The yard houses at 20–50 m are resolution-limited at 480 px: the pixel footprint is ≈ τ at 40 m, and
GT depth fed into fusion reaches only 14 % completeness beyond 40 m at 480 px and 21 % at 960 px (yard
C overall: 480 → 960 px 0.461 → 0.526). Hence the presets keep 800–1600 px. More keyframes help most
in large scenes: with GT depth, 60 instead of 30 yard views raises C from 0.461 to 0.570. A 480 px
object is already saturated.

### Open

* Low-texture walls in the object room stay sparse.
* The yard TSDF mesh loses precision to the noisier far points.
* Not tested: `topK`, the ACMH thresholds, the perturbation schedule, `sigmaColor`, window size, and
  relaxing source selection at trajectory turns (yard frame 184 keeps 3 sources).

## Sky / low-texture halo removal

### Problem

With the tuned presets, sky-coloured points on roof and tree outlines were 27 % of the yard points
farther than τ (5,722 of 21,021). The halo vote (`depthHaloMask`, radius 8) cannot remove a band wider
than its window: inside the band, most same-coloured samples are band pixels on the same extended
plane. PatchMatch's own texture gate (`minVariance`, bilateral weights with σc 3) does not fire either,
because a sky pixel next to a roof keeps small but non-zero weights on the high-contrast roof samples.

### Setup

Same harness, poses, 480 px, 30 / 24 views and `--preset balanced` as above. The yard maps were
recomputed with `--save-maps` and reproduce the tuned result exactly (P 0.9809, C 0.3523). Every row
re-fuses the saved maps with one filter change.

* Sky-coloured points: `b ≥ 150`, `b − r ≥ 20`, `b ≥ g` (the yard sky runs from (205, 218, 232) to
  (95, 140, 205)).
* "Object whole scene" = P / C on the base GT set without the ROI. The object's metrics proper cover
  only the table ROI and the focus set, so they cannot see the room's untextured walls.

### Texture measure (`segmentTextureSampler`)

Grey std of the window samples whose colour is within `segmentColor` of the centre pixel, on the
undistorted frames:

| pixels | std < 1 | std < 1.5 | std < 2 |
|---|---|---|---|
| yard sky | 68–83 % | 85–93 % | 94–96 % |
| yard surfaces | ≤ 0.1 % | ≤ 0.7 % | ≤ 1.7 % |
| object room (plaster walls) | 14–16 % | 30 % | 38 % |

Texture alone cannot separate the sky from the object room's walls. Fused sky pixels are harder: only
59 % have std < 2 and are top-connected, because pixels on the blurred outline measure std 2–6 — hence
the threshold of 3.

### Region coherence

Top-connected untextured regions (std < 2) on the PatchMatch geometric maps; coherence = fraction of
pixels with a valid depth that has ≥ 3 of 4 neighbours valid and on its plane:

| regions | coherent / area | coherent / valid | valid / area |
|---|---|---|---|
| yard sky (39 regions, 688 k px) | 0.1–0.5 (100 %) | < 0.5 for 95 %, ≤ 0.8 | 0.5–1 |
| object room walls (78 regions, 965 k px) | ≥ 0.5 for 99.5 % | ≥ 0.7 for 99.5 % | 0.3–1 |

PatchMatch fills the sky with locally incoherent depths, a wall gets one plane. The rule uses
coherent / area, not coherent / valid: a sky that PatchMatch leaves mostly invalid (likely on denoised
phone images) would contain only the coherent halo bands, and would look coherent per valid depth.

### Experiments

Yard P / C, sky-coloured points > τ, all points > τ | object P / C | object whole scene P / C:

| # | Change | yard | object | object whole scene |
|---|---|---|---|---|
| 0 | reference (tuned presets, halo vote r 8) | .9809 / .3523, 5722, 21021 | .9997 / .9874 | .9557 / .3824 |
| 1 | drop every untextured pixel (std < 1.5) | .9849 / .3520, 1291, 16490 | .9997 / .9874 | .9843 / .2741 |
| 2 | drop untextured pixels connected to the top row: std < 1.5 · 2 (object also std < 1) | .9844 / .3522, 1829, 17117 · .9850 / .3521, 1163, 16444 | .9997 · .9998 / .9874 | .9833 / .2881 · .9879 / .2700 (std < 1: .9599 / .3707) |
| 3 | 2 (std < 2) but a wide same-colour plane vote instead of dropping: r 16 s 4 · r 24 s 6 · r 32 s 8 | .9838, 2521 · .9847, 1457 · .9849, 1259 (C .3521–.3522) | .9998 / .9874 | .9591 / .3770 · .9634 / .3695 · .9675 / .3620 |
| 3b | 3 with invalid samples ignored: r 24 · r 32 | .9844, 1763 · .9848, 1359 | .9998 / .9874 | .9609 / .3757 · .9645 / .3696 |
| 4 | larger halo window for every pixel: r 16 s 4 · r 24 s 6 | .9855 / .3443, 2119, 15705 · .9888 / .3328, 864, 11947 | .9999 / .9877 · .9999 / .9874 | .9594 / .3745 · .9643 / .3620 |
| 5 | fusion: untextured (std < 2) reference pixels need 3 · 4 views | .9840 / .3520, 2407 · .9847 / .3519, 1597 | .9997 / .9874 (3) | .9834 / .2966 (3) |
| 6 | **sky test** (std < 2, step 1): drop all depths of top-connected untextured regions with coherent / area < c; c .2 · .35 · **.5** · .7 | .9823, 4109 · .9847, 1425 · .9850 / .3521, 1163, 16444 · = .5 | .9997 / .9874 | unchanged for c ≤ .5; c .7: .9655 / .3415 |
| 7 | 6 (c .5) with std < 1.5 · 3 · **3, step 2** · 4, step 2 · 2, step 2 | .9844, 1829 · .9855 / .3520, 572, 15838 · **.9854 / .3521, 707, 15985** · .9854, 672 · .9850, 1103 | .9997 / .9874 | .9558 / .3822–.3823 |

Kept: row 7 (std < 3, texture step 2, coherence 0.5) as the sky test of `depthHaloMask` /
`removeDepthHalos`, on by default. Step 2 gives the same result as step 1 at a fifth of the sampling
cost.

Not kept:

* Rows 1, 2 and 5: they remove the object room's walls, −8.6 to −11 pt whole-scene completeness. Row 5
  is also impractical in the engine, because fusion only gets colour thumbnails.
* Rows 3 and 3b: −0.5 to −2 pt object wall completeness for yard gains that row 7 matches without that
  loss.
* Row 4: −0.8 / −1.95 pt yard completeness; the wider window erodes every textured edge, not just the
  sky.
* Coherence 0.7: object walls −4 pt.
* A PatchMatch-side gate was not tried. A segment texture test in the NCC rejects the same pixels as
  row 1 (the reference image alone decides), so it would inherit row 1's wall loss and need a GPU/CPU
  kernel change.

### Final, sky test off → on (balanced, 480 px, saved maps, with TSDF)

| metric | yard | object |
|---|---|---|
| precision | 0.9809 → **0.9854** | 0.9997 → 0.9997 |
| completeness | 0.3523 → 0.3521 | 0.9874 → 0.9874 |
| F | 0.5184 → 0.5188 | 0.9935 → 0.9935 |
| completeness, subset visible | 0.4234 → 0.4232 | 0.9917 → 0.9917 |
| points farther than τ | 21,021 → 15,985 (−24 %) | 192 → 192 |
| sky-coloured points farther than τ | 5,722 → 707 (−88 %) | 0 → 0 |
| whole scene P / C | .9712 / .3036 → .9784 / .3034 | .9557 / .3824 → .9558 / .3822 |
| mesh P / C / F | .845 / .347 / .492 → .868 / .346 / .495 (accuracy p90 0.145 → 0.124 m) | .9989 / .9760 / .9873 → .9987 / .9751 / .9868 |

* Yard precision by distance to the nearest camera: 10–20 m 97.9 → 98.6 %, 20–40 m 88.3 → 91.2 %,
  > 40 m 53.3 → 58.0 %. Completeness unchanged in every distance bin (±0.1 pt).
* Yard fusion-input pixels without GT (sky) removed: 80.9 → 94.9 %. Pixels within 1 % removed:
  0.77 → 0.82 %.
* Object: 99 of 757 k points removed; ROI metrics identical.
* The viewer render of yard frame 120 no longer shows the light-blue rim around roofs and trees.
* Remaining yard error: far-field 2-view depth noise. 707 sky-coloured points remain, out of 15,985
  farther than τ.

Cost: +7–17 ms per 480×360 map on one x86 core (yard frame 120 34 → 44 ms, object frame 80 with a
large wall region 38 → 54 ms), scaling with the top-connected untextured area. Transient memory
+9 B/px. The halo vote itself: 37 ms.

### Orientation (`skyUp`)

Measured on the saved maps (balanced, 480 px); `haloRotate=k` in the harness runs the filter on views
rotated by k quarter turns, `haloSkyUp=true` passes each view's GT up direction:

| views | yard P / C, sky-coloured > τ, all > τ | object P / C, whole scene P / C |
|---|---|---|
| upright, default (top row) | .9854 / .3521, 707, 15,985 | .9997 / .9874, .9558 / .3822 |
| upright, `skyUp` from the pose | identical | identical |
| rotated 90°, no `skyUp` | .9829 / .3522, 3,446, 18,734 | .9997 / .9874, .9558 / .3824 |
| rotated 180°, no `skyUp` | .9809 / .3523, 5,722, 21,021 (= sky test off) | .9997 / .9874, .9557 / .3824 |
| rotated 90° · 180°, with `skyUp` | identical to upright | identical to upright |

At 90° the sky still reaches the top row at one corner, so part of it is found; at 180° nothing is.

### Phone-realistic captures

`yard-real` / `object-real` (JPEG 0.85, motion blur, white-balance drift, focus breathing, rolling
shutter; e2e/README "Real-capture artefacts"). Mini subsets with GT poses: yard 15 views, object 12
views, stride 8, balanced, 480 px, re-fused with the sky test off and on. The low absolute P / C come
from the capture artefacts under GT poses, not from the halo filter.

| | sky test off | sky test on |
|---|---|---|
| yard-real P / C / Csub | .9129 / .0994 / .2894 | .9153 / .0993 / .2890 |
| yard-real sky-coloured > τ, all > τ | 1,742, 44,584 | 405 (−77 %), 43,238 |
| yard-real fusion-input pixels without GT removed | 82.3 % | 94.0 % (≤ 1 % pixels: 1.6 → 1.6 %) |
| object-real P / C / Csub (ROI) | .9995 / .7409 / .9100 | identical (84 of 300 k points removed) |
| object-real whole scene P / C | .8719 / .0232 | .8736 / .0230 |
| object-real fusion-input pixels removed: > 5 % · 1–5 % · ≤ 1 % | 50.6 · 7.5 · 0.7 % | 52.4 · 10.2 · 0.9 % |

* The texture gate holds up under JPEG 0.85 on yard-real: of the fusion-input sky pixels (no GT),
  72.7 % have std < 2 (clean yard 79.7 %) and 79.1 % std < 3. All 22 top-connected untextured regions
  are GT sky, each with coherence < 0.7.
* The coherence margin shrinks on object-real: 19 % of the top-connected untextured wall area (78 k of
  414 k px) falls below coherence 0.5 (clean object: 0.5 %) and is treated as sky. The artefacts mostly
  ruin those wall depths anyway: these pixels hold 45 % of the > 5 % outliers but only 1.1 % of the
  ≤ 1 % pixels (clean object at std < 2: 16 % and 14.5 %). Net: +0.17 pt whole-scene P for −0.02 pt C.
  This is the first case to re-check on device; if real walls keep good but less coherent depths, lower
  `skyMinCoherence` (at 0.35 the clean yard keeps 1,425 sky-coloured points farther than τ instead of
  707, row 6).

## Dense on a real GPU

### Setup

`E2E_GPU=host` (headless Chrome 155, Apple M4 10-core GPU, Metal), host shared with other heavy jobs
(load 5–20, wall times ±30 % and more). GPU time from timestamp queries
(`EngineConfig.denseGpuTimestamps`, `PatchMatchGPU.create(…, { timestamps: true })`,
`lastTimings.gpuMs`), which stay meaningful under load. The engine logs the wall-time split of the
depth-map and fusion stages. Full e2e tables: e2e/README.md "Dense on the real GPU".

### Where the time went (before the changes below)

yard-real `fast`, 40 views at 800×600, load 6–8: depth maps 38.1 s = view loads 10.4 s (83 JPEG
decodes + undistort + rolling-shutter warp, ~125 ms each), PatchMatch 14.5 s (upload 3.5, GPU passes
7.5 of which 5.5 s GPU execution, readback 3.5), halo filter 6.0 s, IndexedDB store 5.7 s, pass inputs
1.4 s; fusion 8.3 s. On a real GPU the kernel is a minority of the stage; at `balanced` (object-real,
80 views 720×960) it is half: depth maps 154.9 s with 47.8 s GPU execution.

### Point cap: footprint-adaptive LOD (`footprintDownsample`)

The previous cap merged the cloud with one voxel size from the median radius of the first 100 k points
(dominated by the room / far field): object-real `fast` 3.29 M fused → 0.57 M points, accuracy p50 / p90
0.28 / 0.86 mm (cap lifted) → 0.44 / 1.8 mm; yard-real precision 0.945 → 0.80, accuracy p50 8.3 → 25 mm.

Offline on the same cap-lifted clouds (cap 2 M):

| | footprint LOD | global voxel | no cap |
|---|---|---|---|
| object p50 / p90, P, C | 0.29 / 0.93 mm, 0.9997, 0.9969 | 0.51 / 1.8 mm, 0.9982 | 0.28 / 0.86 mm, 0.9998, 0.9969 |
| yard p50, P / C / F | 8.9 mm, 0.931 / 0.390 / 0.550 | 26 mm, 0.781 / 0.402 / 0.531 | 8.3 mm, 0.945 / 0.405 / 0.567 |
| balanced yard (6.3 M fused, cap 4 M) P / C / F, p50 | 0.953 / 0.484 / 0.642, 7.1 mm | 0.898 / 0.494 / 0.637, 13.5 mm | 0.962 / 0.494 / 0.653, 7.1 mm |

The global voxel's slightly higher C comes from the far field: its cells (2.3 cm there) are far below a
40 m footprint, so it merged the near field ~10× and left the far points alone. Keeping the medoid
instead of the mean changes nothing measurable. 1.8–3.9 s for 3.3–6.3 M points in the e2e worker.

### Multi-scale

½-resolution photometric (`iterations`) → full-resolution prior-initialised photometric
(`fullResIterations` 2) → geometric. GT-pose harness (fast, 800 px, 40 views; GPU s / P / C / F):
object 9.0 → 5.8 s, 0.9999 / 0.9985 / 0.9992 → 0.9998 / 0.9981 / 0.9990; yard 4.5 → 3.7 s, 0.9930 /
0.4490 / 0.6184 → 0.9928 / 0.4497 / 0.6190. With only 4 photometric iterations (`fast`) the GPU saving
is small in the engine (object-real 7.8 → 6.6 s, yard-real 5.5 → 4.7 s); `balanced` (6 + 3 iterations)
object-real 47.8 → 39.8 s.

### CPU-side measurements

* Native `Float16Array` conversion for float16 depth maps / clouds: 5–9× faster than the scalar
  fallback, bit-identical; the IndexedDB store of a map ~35 % cheaper.
* Letting the image LRU grow into what the PatchMatch stage leaves below the fusion / mesh peak: each
  view decoded once per resolution (yard-real fast 83 → 80 decodes for three passes instead of two;
  object-real balanced 210 → 160).
* `Math.hypot` is 4–5× slower than `Math.sqrt` in V8 (`planesToDepthMap`).

### Kernel diet (`affineWindow`, opt-in, not in the presets)

`affineWindow` + `patchRadius` 6 / `patchStep` 3 (symmetric 5×5 = 25 samples over 13 px) +
`numSamplesPerIter` 3 halves the GPU time again (e2e fast, with multi-scale: object-real 7.2 → 3.9 s,
yard-real 6.3 → 3.1 s) but costs yard quality: e2e yard-real P / C / F 0.942 / 0.401 / 0.563 → 0.932 /
0.396 / 0.555, accuracy p50 8.0 → 9.8 mm; clean yard (GT harness) C 0.4503 → 0.4342. Each part alone
(affine + 3 proposals at 11 px, the 9 px / 25-sample window, the 13 px window with 6 proposals) costs
0.8–1.0 pt of clean-yard completeness. Object: equal within noise (object-real 0.29 / 0.98 mm → 0.32 /
1.1 mm, C 0.9969 → 0.9981). At `balanced` the cost is smaller: yard-real GPU 31.0 → 15.4 s (densify
125.9 → 111.2 s), P / C / F 0.953 / 0.483 / 0.641 → 0.946 / 0.483 / 0.639, p50 7.3 → 7.8 mm — a
candidate for balanced / max after an on-device check.

### Results (e2e, worker engine, back-to-back, same load)

Densify wall s (depth maps / fusion) and dense quality, before → after the changes above:

| run | densify s | quality |
|---|---|---|
| object-real fast | 60.7 (47.3 / 12.3) → 51.1 (37.7 / 12.2) | p50 / p90 0.44 / 1.8 → 0.29 / 0.99 mm |
| yard-real fast | 47.4 (38.1 / 8.3) → 36.4 (26.8 / 8.8) | P / C / F 0.803 / 0.408 / 0.541 → 0.935 / 0.397 / 0.557, p50 25 → 8.8 mm (`denseAccuracyMedian` FAIL → PASS) |
| object-real balanced | 181.0 (154.9 / 24.8) → 148.3 (122.2 / 24.7) | 0.37 / 1.4 → 0.28 / 0.87 mm |
| yard-real balanced | 148.6 (127.8 / 19.7) → 125.9 (105.0 / 19.7) | P / C / F 0.898 / 0.493 / 0.636 → 0.953 / 0.483 / 0.641, p50 13.7 → 7.3 mm |

Completeness of clouds above 3 M points is scored on every point (`e2e/lib/metrics-core.mjs`).

### Remaining

The CPU stages now dominate the depth-map stage: halo filter (~150–250 ms per 0.5 MP map, on the main
path between GPU passes), decode + undistort + rolling-shutter warp (~90 ms per view, two bilinear
resamplings), the IndexedDB store, and CPU fusion (7–25 s). Candidates: run halo / store in a helper
worker or overlap them with the next view's GPU passes (needs a cooperative halo filter), undistort +
RS warp in one GPU resampling, GPU fusion. The error-scope round trip of `compute()`'s set-up and the
~100 band submissions per call cost 20–30 % over the GPU execution time under contention. Timings on
Apple mobile GPUs (iPhone) are still unmeasured.
