# Fly-through video and SPZ export: format checks and first measurements (2026-10-22)

These are the measurements behind src/io/mp4.ts, src/app/video-export.ts, the offscreen frames of
src/viewer/splat-layer.ts and src/splat/spz.ts. They were taken in a Linux container with
headless Chromium (Playwright 1.56, SwiftShader WebGPU) and ffmpeg 6.1. **No iPhone or host-GPU
measurement yet** (see the end).

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
* Playwright's Chromium has no H.264 encoder: `isConfigSupported` rejects avc1.640028 / 4d0028 / 42e01f.
  With a real VP8 VideoEncoder in its place, the same loop encoded 30 frames at 128×192 (one raw
  frame alive at a time). The decoded frames scored 41.3–41.5 dB PSNR against the directly read
  frames. SwiftShader takes 169 ms of GPU time per frame there (software rasteriser).

## Still to measure

* iPhone (iOS 26): Diagnostics → "Test video encoding". Record the H.264 codec chosen (portrait and
  landscape), the frame path ('canvas' vs 'readback'; WebKit's VideoFrame-from-WebGPU-canvas is
  unverified), ms per 720p frame and the decode check. Then record a real export's ms per frame from
  the console line `[video] …`.
* Host GPU (M4, Chrome, H.264): `E2E_GPU=host node e2e/ui/flythrough.mjs <run>` on object-real and
  yard-real `--splat` runs: 180–300 frames, decoded vs direct PSNR ≥ 34 dB, ≤ 30 s for 300 frames,
  peak memory.
* SPZ in third-party viewers (SuperSplat, Spark): orientation and colours by eye, and `nvs.mjs`
  scoring of the decoded SPZ against splats.ply (≤ 0.3 dB held-out).
