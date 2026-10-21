# Viewer rendering and picking: measurements

Measurements from 2026 (up to 2026-10); the code may have changed since — check the source before relying on a number.

Code: `src/viewer/` (three.js WebGL2 viewer, EDL, picking) and `src/viewer/splat-layer.ts` (WebGPU Gaussian splat layer).

## Picking accuracy and cost

| What | Result | Source |
|---|---|---|
| GPU pick (two-pass depth readback + planar fit of 1/z), 240×180 px test canvas, surface viewed at 30° | ≈ 5 mm error at 7 m (≈ 0.1 px) | `viewer.gpu.test.ts` |
| Picks / measures at 100 m scale with km-distant outliers | < 0.5 % distance error | `viewer.review.gpu.test.ts` |
| Measuring on hidden points while the splats are shown | 1.0438 measured vs 1.0440 true (test asserts < 1 %) | `splat-layer.gpu.test.ts` |
| CPU picker (`pickNearestProjected`), points | ≈ 10–30 ms per 2 M points, one core | `helpers.ts` |
| CPU picker meshes (`raycastTriangles`, Möller–Trumbore) | ≈ 10–20 ms per 1 M triangles, one core | `helpers.ts` |

The GPU path costs 1 + 1 per tap small renders and ≤ 64×64 px readbacks (a few ms), independent of the point count.

## Splat camera

`splatCameraFromView` vs three's own projection: < 1e-3 px over 40 random cameras / world roots (`splat-camera.test.ts`). GPU test: Gaussian centroids landed on `worldToScreen` of their means within 0.06 px (the test asserts < 0.15 px; `splat-layer.gpu.test.ts`).

## Splat layer frame time and memory

Harness: `e2e/ui/splat-shots.mjs`, `E2E_GPU=host`, Apple M4 / Metal, iPhone emulation 393×852 @3x → splat canvas 786×1704 = 1.34 MP (DPR capped by `maxPixelRatio` 2 and the 2 MP budget), 200 k Gaussians, host load 5–8.

| Scene | Intersections | Full-size frame while orbiting, submit → done (median / p90) |
|---|---|---|
| object-real | 0.93 M | 16.3 / 22.5 ms |
| yard-real | 0.34 M | 12.7 / 18.1 ms |

Splat layer GPU memory: 107 MB, of which the intersection reservation (3.2 M) is 51 MB and the per-pixel buffers 27 MB.

No iPhone measurement yet. A full frame there is expected to take several times longer; that is why frames drawn while the view moves are capped at 0.5 MP (when the last full frame took > 16 ms) and upscaled.

While a rebuild trains a new splat model, the viewer frees the previous model's layer: ≈ 150–220 MB of GPU memory at 300–500 k Gaussians that the engine's page reserve does not cover.

## Unlit mesh default

Mesh vertex colours are photo colours that already contain the shading; lighting them shades them twice. Covered PSNR in held-out views (e2e/README.md "Baselines"):

| Scene | Lit | Unlit |
|---|---|---|
| object-real | 23.6 dB | 28.7 dB |
| yard-real | 23.1 dB | 26.9 dB |

Hence `DisplayOptions.unlit` defaults to true. Points keep EDL.

## Framing for splats

Splats are photoreal near the photos and show unconstrained floaters / the sky shell far outside the capture: the from-above overview of yard-real is a blur (e2e/README.md "Viewer splat layer"). This is why `viewFromCamera` (look along a capture camera at the median content depth, pulled back 15 %) exists and is used to open scans with splats.
