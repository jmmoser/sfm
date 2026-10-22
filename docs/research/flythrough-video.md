# Fly-through video and SPZ export: format checks and first measurements (2026-10-22)

These are the measurements behind src/io/mp4.ts, src/app/video-export.ts, the offscreen frames of
src/viewer/splat-layer.ts and src/splat/spz.ts. They were taken in a Linux container with
headless Chromium (Playwright 1.56, SwiftShader WebGPU) and ffmpeg 6.1. **No iPhone or host-GPU
measurement yet** (see "Still to measure").

## MP4 writer against ffmpeg

Two libx264 clips, 720×1280 at 30 fps with 120 frames: High with 2 B-frames, and Baseline. Each
was read with `readMp4` and written again with `Mp4Writer` (avcC copied, the samples in decode
order, timestamps from the source's pts). ffmpeg decodes both rewritten files without errors. The
frame-md5 sequences are identical to the originals'. ffprobe reports 30/1 fps, 4.000 s and pts
from 0 in 1/30 s steps; for the B-frame clip that comes from ctts plus the edit list.
Reproduce: `ffmpeg -f lavfi -i testsrc2=size=720x1280:rate=30:duration=4 -c:v libx264 -bf 2 -g 60
src.mp4`; remux it in Node through Vite's `ssrLoadModule('src/io/mp4.ts')`; then compare
`ffmpeg -i <file> -f framemd5 -` of both files.

## SPZ v3 against the reference loader

The reference C++ library (github.com/nianticlabs/spz, main, 2026-10-21) was built from
`load-spz.cc` and `splat-types.cc` against a ZSTD stub (`g++ -std=c++17 … -lz`, with a small
`main` that calls `spz::loadSpz` and prints the cloud); the v1–3 gzip path does not use ZSTD. It
loaded files written by `encodeSplatSPZ` with `UnpackOptions.to = RFU` (the app's Z-up world),
300 random Gaussians each:

| SH degree | position | log-scale | alpha | f_dc | f_rest | rotation |
|---|---|---|---|---|---|---|
| 0 | 7.6 µm | 0.031 | 0.0019 | 0.013 | – | 0.16° |
| 1 | 7.6 µm | 0.031 | 0.0020 | 0.013 | 0.035 | 0.17° |

These are the largest errors against the model. Each is the format's quantisation step or half of
it: 2^−17 m at 16 fractional bits, 1/32, 1/510, 0.5/38.25, and the 5-bit SH buckets. The load
includes the reference's own RUB → RFU conversion, which also rotates the SH coefficients.

On a smooth 3 600-Gaussian surface model at SH 0, the SPZ is 6.8 B per Gaussian, 10.0 % of the
INRIA PLY. Random Gaussians barely compress: they come out at about 20 B each, 29 % of the PLY.

## Offscreen splat frames and the encode loop (headless Chromium, SwiftShader)

* `SplatOffscreen.pixels` is identical to `SplatRenderer.render` of the same camera: max |Δ| 0 at 96×64.
* `new VideoFrame(OffscreenCanvas 'webgpu')` holds the rendered image in Chromium (format BGRX;
  the PSNR check passes), so the 'auto' session picks the canvas path. The probe's cleared-texture
  check agrees.
* Snapshot checks (added after review): the 'auto' session's first two frames are compared with
  their readback after the canvas has been cleared to magenta, and a new canvas frame must then be
  magenta. Chromium passes both: the VideoFrame is a copy taken at construction, not a view of
  the texture. The Diagnostics probe does the same with two clear colours, and checks every
  decoded frame's colour (consecutive probe frames differ by ≥ 89 levels). With a stand-in codec
  that repeats the first or the previous frame, the probe reports the frame that went wrong
  (src/app/video-probe.gpu.test.ts). This is the check the iPhone run needs: WebKit's
  `makeRenderingResultsAvailable` path is still unverified.
* Playwright's Chromium has no H.264 encoder: `isConfigSupported` rejects avc1.640028 / 4d0028 / 42e01f.
  With a real VP8 VideoEncoder in its place, the same loop encoded 30 frames at 128×192 (one raw
  frame alive at a time). The decoded frames scored 41.3–41.5 dB PSNR against the directly read
  frames. SwiftShader takes 169 ms of GPU time per frame there (software rasteriser).

## Still to measure

* iPhone (iOS 26): Diagnostics → "Test video encoding". Record the H.264 codec chosen (portrait and
  landscape), the frame path ('canvas' vs 'readback'; WebKit's VideoFrame-from-WebGPU-canvas is
  unverified; a note says which snapshot check failed), ms per 720p frame and the decode check
  (frame count, every frame's colour). Then record a real export's ms per frame and frame path
  from the console line `[video] …`.
* Host GPU (M4, Chrome, H.264): `E2E_GPU=host node e2e/ui/flythrough.mjs <run>` on object-real and
  yard-real `--splat` runs: 180–300 frames, decoded vs direct PSNR ≥ 34 dB, ≤ 30 s for 300 frames,
  peak memory.
* SPZ in third-party viewers (SuperSplat, Spark): orientation and colours by eye.
* SPZ quality (≤ 0.3 dB held-out below splats.ply): a `run-e2e.mjs --splat` run writes splats.spz
  next to splats.ply (the `spz` export is in the default export list). Score both:
  `node e2e/nvs.mjs <dataset> <run> --renderers splat` (→ `<run>/nvs/`) and
  `node e2e/nvs.mjs <dataset> <run> --renderers splat --splat-file splats.spz` (→ `<run>/nvs-spz/`).
  Not run yet: no splat run in this container.

## Camera path on synthetic captures (after review)

The first path builder failed on captures the README asks for. An out-and-back walk with a 180°
turn on the spot became 10 s of turning: the turn limit was taken from the window that the limit
itself selects, so it fell to 10 °/s, and the window was then centred on the turn. Two-height
orbits were averaged across the rings (up to 17.7 cm off the photos). A top-down shot and an
azimuth gap made the orbit sweep azimuths that no photo covers. The rebuilt builder (header of
src/viewer/flythrough.ts) fixes the turn limit once for the whole path, shortens pans of an
over-long path to a quarter of the clip, and crops to the window with the most translation. In
object mode it replays one azimuth pass. `src/viewer/flythrough.test.ts`, 2026-10-22:

| capture | mode | off path | view | linear step ratio | notes |
|---|---|---|---|---|---|
| out and back, 20 m legs (depth 8 m) | scene | 1.9 cm | 7.2°* | 1.00 | 13.6 m of walking shown (before: 0.09 m), 10 % of frames turning or easing |
| out and back, 4 m legs (turn: 1 cm jitter or exact centres) | scene | 1.9 cm | 7.7°* | 1.00 | whole capture, the 180° turn eased in ≈ 2.9 s (27 % of frames) |
| walk, 90° pan in the middle (scene) | scene | 1.8 cm | 7.9°* | 1.00 | 13.3 m shown (before: a 9 s pan, linear step ratio 86) |
| two rings, same direction | object | 2.7 cm | 6.3° | 1.00 | before: 5.9 cm |
| two rings, back and forth | object | 2.4 cm | 5.4° | 1.00 | one ring (and the rise it starts from) |
| 180° arc + top-down shot | object | 1.3 cm | 16.5° | 1.00 | no frame > 8° of azimuth from a photo |
| 0–90° and 180–300° (gap) | object | 0.7 cm | 4.7° | 1.00 | the 120° arc only |

View: the largest angle to the nearest keyframe's view; * to the nearest keyframes' (within
0.25 m of the nearest), since an out-and-back walk passes each spot twice, looking both ways.

The e2e poses still pass: object-real 0.9 cm / 10.2° / angular step ratio 1.35, and yard-real
13.6 cm / 17.7° / 1.92.
