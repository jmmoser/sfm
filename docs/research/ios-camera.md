# iOS Safari camera capture for on-device photogrammetry (iOS 26 / iOS 27)

Research date: 2026-10-16. Shipping versions at that date: **Safari 26.6** (iOS 26.6, 2026-07-27), **Safari 27.0** (iOS 27.0, 2026-09-14). **Safari 27.2 beta** came out 2026-10-13. ([Safari release notes index][rn-index])

How to read the evidence tags:

| Tag | Meaning |
|---|---|
| **[RN]** | Apple Safari release notes for that version. |
| **[WK]** | WebKit source on `main`, fetched 2026-10-16. `main` is ahead of what ships. A [WK]-only fact is very likely true in Safari 27.x, but that has not been proven for each version. |
| **[BCD]** | MDN browser-compat-data (`main`). |
| **[AVF]** | Apple AVFoundation documentation. |
| **[3P]** | A third-party measurement or library. |
| **UNVERIFIED** | No primary source or on-device test confirms it. In those cases I give a best estimate and a fallback. |

---

## 0. Key conclusions

1. **Do not rely on `facingMode: "environment"` alone.** On multi-lens iPhones it selects a *virtual* camera: "Back Triple Camera" on Pro models, "Back Dual Wide Camera" on dual-lens models. These cameras **switch physical lenses automatically** based on focus distance (macro), zoom and light. Every switch silently changes the intrinsics. To prevent this, choose the physical **"Back Camera"** (`AVCaptureDeviceTypeBuiltInWideAngleCamera`) by `deviceId`. ([WK][wk-avvcs], [bug 250142][bug250142], [bug 253186][bug253186], [Apple forum 776460][forum776460])
   - Selecting the physical wide camera by `deviceId` should prevent lens switching. That has not been confirmed on a device (**UNVERIFIED**). A physical `AVCaptureDevice` has only one lens. The forum report does not say which `deviceId` it used, and it describes switching among three lenses, which only a Triple virtual camera can do.
2. **The resolution list is the camera's native AVFoundation formats. WebKit adds no cap.** On the wide camera the maximum is typically **4032×3024 (4:3) or 3840×2160 (16:9) at ≤30 fps**. If you set no constraints you get **640×480 at 30 fps**. ([WK][wk-avvcs], [WK MediaConstraints][wk-mc])
3. **Use a 4:3 format. 1920×1440 is the default choice; use 4032×3024 for maximum quality.** On the only device with a published format dump (iPhone 12 Pro Max), these formats keep the full-sensor FOV (≈67° horizontal on a 26 mm-equivalent main camera). On 48 MP quad-Bayer phones (14 Pro and later), whether the *first* 1920×1440 format is the full-FOV, non-binned one is **UNVERIFIED**. The web cannot read `videoFieldOfView`, so measure the FOV with a calibration (§8). 16:9 formats crop the sensor vertically. Some low-resolution and binned formats crop it further. ([3P dump][zenn], [AVF archive][avf-archive])
4. **By default, pixel buffers are not rotated; rotation is metadata only.**
   - In a portrait UI, `getSettings()` and `<video>.videoWidth/Height` report swapped values, for example 1440×1920.
   - `MediaStreamTrackProcessor` (MSTP) `VideoFrame`s keep the sensor-landscape `codedWidth×codedHeight`. Safari has no `VideoFrame.rotation`. This is good for SfM, because the intrinsics stay fixed. ([WK][wk-avvcs], [WK VideoFrameCV][wk-vfcv], [BCD])
   - **Exception:** WebKit rotates the pixels for every consumer of a source once any consumer of that source asks for it. `MediaRecorder` and the WebRTC sender (`RealtimeOutgoingVideoSource`) can ask. Never attach those to the track you feed to MSTP. Also assert `codedWidth > codedHeight` on every frame. ([WK RealtimeMediaSource][wk-rms], [WK MediaRecorderPrivateAVFImpl][wk-mravf])
5. **Frame access path:** clone the track, transfer it to a Dedicated Worker, then read it with `MediaStreamTrackProcessor` (Safari 18+, worker-only, default `maxBufferSize` 1). Frames arrive as NV12 full-range, and the Y plane can be used directly. `VideoFrame.timestamp` is the capture presentation timestamp in µs. ([RN 18][wk-18], [BCD], [WK][wk-mstp])
6. **Settable controls:** `zoom`, `torch`, `whiteBalanceMode` (manual/single-shot/continuous), `frameRate`, and the non-standard `powerEfficient`.
   - **Not supported:** `focusMode`, `focusDistance` (it appears in capabilities only), `exposureMode`, `exposureCompensation`, `exposureTime`, `iso`, `pointsOfInterest`, `resizeMode`. ([WK IDL][wk-mtc])
   - **Stabilization:** WebKit never enables video stabilization; the AVFoundation default is *off*.
   - **Distortion correction:** geometric distortion correction is *on by default* wherever AVFoundation supports it (the ultra-wide camera and virtual devices that contain it). The web page cannot control it. ([AVF][avf-stab], [AVF][avf-gdc])
7. **Page hidden means capture is interrupted.** On iOS, `InterruptVideoOnPageVisibilityChangeEnabled` is true: the track fires `mute` when the page is hidden and `unmute` when it is visible again. ([WK prefs][wk-prefs])
   - Screen Wake Lock works: Safari 16.4+, and inside Home Screen apps from 18.4.
   - `screen.orientation.lock()` is **not available**. ([BCD], [WK prefs][wk-prefs])
8. **None of these are exposed to the web on iPhone:** LiDAR, depth, TrueDepth, WebXR/ARKit poses, manual exposure and manual focus. Pose has to come from SfM plus IMU.

---

## Summary table

| Capability | iOS Safari status (26.x / 27.x) | Since | Evidence |
|---|---|---|---|
| `getUserMedia` (HTTPS only) | Yes | 11 | [BCD] |
| Available sizes | Every distinct size in the selected camera's `AVCaptureDevice.formats`. The first format of each size wins. | — | [WK][wk-avvcs] |
| Largest video size (wide camera) | Typically **4032×3024 @ ≤30 fps**; 3840×2160 @ 30 fps. Above 4032 on 48 MP sensors: **UNVERIFIED (unlikely)**. | — | [WK], [3P][zenn] |
| Default with no constraints | 640×480 ideal, 30 fps ideal | — | [WK][wk-mc] |
| `getSettings()` width/height | Swapped when the UI is rotated ±90° (portrait on iPhone) | — | [WK][wk-avvcs] |
| Pixel rotation | Metadata only. `<video>` displays rotated; MSTP frames keep sensor orientation. | — | [WK] |
| `enumerateDevices` labels | Localized `AVCaptureDevice.localizedName`. Only appear after permission is granted. | — | [WK][wk-avcdm], [3P][scandit] |
| `facingMode:"environment"` choice | Triple > DualWide > UltraWide > Dual > Wide > Tele, so a lens-switching virtual camera wins | 16.4+ | [WK][wk-avvcs], [bug 253186][bug253186] |
| `zoom` | Settable. On Triple/DualWide it is normalized so 1 = wide and 0.5 = ultra-wide. Max ≤ 10. | 17 (261679@main) | [WK], [bug 253646][bug253646] |
| `torch` | Settable when the camera has a torch | ≈17 (working on 17.5.1); `getSettings` fixed Dec 2024 | [bug 243075][bug243075], [bug 280970][bug280970] |
| `whiteBalanceMode` | Settable: manual (locked), single-shot, continuous | — | [WK] |
| `focusDistance` | Capability only (minimum focus distance in metres). **Not settable.** | — | [WK][wk-mtc] |
| `focusMode`, `exposureMode`, `exposureCompensation`, `exposureTime`, `iso`, `colorTemperature`, `pointsOfInterest` | **No** (FIXMEs in the IDL) | — | [WK][wk-mtc] |
| `resizeMode` | **No.** WebKit scales silently when the request doesn't match a native size. | — | [BCD], [WK] |
| `frameRate` | Yes. Min and max frame duration are both set to 1/fps (a fixed rate). | — | [WK] |
| `powerEfficient` (non-standard) | iOS only; prefers binned formats | — | [WK] |
| `backgroundBlur` | Read-only reflection of the Portrait video effect | — | [WK] |
| Video stabilization | Off (WebKit never sets it; AVF default is off) | — | [WK], [AVF][avf-stab] |
| Lens distortion correction (GDC) | On where supported (ultra-wide and virtual devices); not controllable | — | [AVF][avf-gdc] |
| `requestVideoFrameCallback` | Yes. For a camera stream: `captureTime`, `presentationTime`, `expectedDisplayTime` (= `presentationTime`), `width`, `height`, `presentedFrames`; `mediaTime` = 0 | 15.4 | [BCD], [WK][wk-mpms] |
| `new VideoFrame(videoElement)` | Yes (clones the current frame) | 16.4 | [BCD], [WK][wk-wcvf] |
| `VideoFrame.rotation` / `flip` / `metadata` | **No** | — | [BCD] |
| `VideoFrame.copyTo({format})` conversion | Present in WebKit `main`; BCD says unsupported. Feature-detect it. **UNVERIFIED.** | ? | [WK], [BCD] |
| `MediaStreamTrackProcessor` | Yes, **DedicatedWorker only**; default `maxBufferSize` = 1 | 18.0 | [RN 18][wk-18], [BCD], [WK][wk-mstp] |
| `VideoTrackGenerator` | Yes (worker) | 18.0 | [BCD] |
| Transferring `MediaStreamTrack` to a worker | Yes (`Transferable`, same agent cluster) | 18.0 (inferred: BCD has no entry, and MSTP 18.0 is worker-only) | [WK IDL][wk-mst-idl], [RN 18][wk-18], [RN 27][rn-27] |
| Transferring `VideoFrame` | Yes (`Serializable`, `Transferable`) | 16.4 | [WK IDL][wk-wcvf-idl] |
| `ImageCapture.takePhoto` / `getPhotoCapabilities` / `getPhotoSettings` | Yes. JPEG at quality 1.0. Window only. Size is effectively the **smallest** `supportedMaxPhotoDimensions` entry of the active format, whatever you request (§5.5). | 18.4 | [BCD], [WK] |
| `ImageCapture.grabFrame` | Yes | 26.0 | [RN 26][rn-26] |
| `createImageBitmap(video)` | Yes | 15 | [BCD] |
| OffscreenCanvas in a worker: 2D / WebGL / WebGPU context | Yes / Yes / Yes | 16.4 / 17 / 26 | [BCD] |
| WebGPU `importExternalTexture(VideoFrame)`, `copyExternalImageToTexture(VideoFrame)`, `navigator.gpu` in workers | Yes | 26.0 | [BCD], [RN 26][rn-26] |
| Maximum canvas area | **8192×8192 = 67.1 MP** on iOS (Safari 18+); 16.7 MP before that. The total-canvas-memory cap was removed in Safari 17. | 18 | [WK 276145@main][wk-canvas8k], [WK 265628@main][wk-canvasmem], [WK][wk-canvasbase] |
| Screen Wake Lock | Yes. Inside Home Screen web apps only from 18.4. | 16.4 / 18.4 | [BCD] |
| `screen.orientation` (`type`, `angle`, `change`) | Yes | 16.4 | [BCD] |
| `screen.orientation.lock()` / manifest `orientation` | **No** (the WebKit setting defaults to false) | — | [BCD], [WK prefs][wk-prefs] |
| Camera while the page is hidden | Interrupted: track is muted, then unmuted on return | — | [WK prefs][wk-prefs] |
| WebXR / ARKit / LiDAR / depth / TrueDepth | **No** | — | [BCD], [WK][wk-avvcs] |

---

## 1. `getUserMedia` on the rear camera: resolutions, frame rates, constraints, settings, orientation

### 1.1 Which resolutions exist

WebKit builds its capture "presets" directly from `[AVCaptureDevice formats]` of the selected camera:

- It keeps one preset per distinct width×height. **The first format of each size wins.**
- It skips only ProRes RAW Bayer formats.
- There is **no WebKit-side resolution cap**. ([WK `AVVideoCaptureSource::generatePresets`][wk-avvcs])

So the list you can get is exactly the camera's native video format list.

The table below shows the measured back wide-angle formats of an **iPhone 12 Pro Max** (26 mm-equivalent; [3P dump][zenn]). It has the same structure as Apple's published tables for the iPhone X and 8 ([AVF archive][avf-archive]).

| Size | Aspect | fps of the *first* format of that size | HFOV (12 Pro Max) | Notes |
|---|---|---|---|---|
| 4032×3024 | 4:3 | 1–30 | 67.10° | Full sensor |
| 3840×2160 | 16:9 | 1–30 | 68.16° | A 60 fps 4K format exists, but the dedup step hides it (**UNVERIFIED** on device) |
| 3264×2448, 2592×1944 | 4:3 | 1–30 | 67.10° | |
| **1920×1440** | 4:3 | 1–30 | **67.10°** | Non-binned, downscaled from 4032. The binned 60 fps variant (63.9°) is hidden. |
| 1920×1080 | 16:9 | 1–30 | 68.16° | The 60/120/240 fps variants are hidden |
| 1440×1080 | 4:3 | 1–60 (binned) | **59.31°** | Cropped |
| 1280×720 | 16:9 | 1–30 | 68.16° | |
| 1024×768, 640×480, 480×360 | 4:3 | 1–60 | **64.72°** | Cropped |

Engineering consequences:

- **Different formats have different sensor crops, and therefore different intrinsics.** Only formats in the same FOV group scale linearly. For example, 1920×1440, 2592×1944, 3264×2448 and 4032×3024 all share 67.10° on that device.
- `getCapabilities()` reports aggregates across presets: `width.max` and `height.max` are the largest preset. Because WebKit can resize, `width.min` is 1. `frameRate.max` is the maximum over *all* presets, so it may say 60 or more even though full resolution runs at 30.
- **48 MP sensors** (14 Pro and later Pro models, 15 and later): 8064×6048 shows up only as a *photo* dimension (`supportedMaxPhotoDimensions`; [3P forum 738418][forum738418]). I expect the largest *video* format to remain 4032×3024. **UNVERIFIED.** Detect it at runtime with `track.getCapabilities()`.

### 1.2 How to request a size

WebKit's matching rules ([WK `bestSupportedSizeFrameRateAndZoom`][wk-rvcs]):

- Presets smaller than the request are ignored.
- An exact size match wins.
- Otherwise WebKit takes a larger preset with the same aspect ratio and downscales it.
- Otherwise it resizes a larger preset to the requested width×height. Whether this crops or stretches is **UNVERIFIED**. Avoid that path.

Sizes are compared against the sensor-landscape preset sizes, whatever the UI orientation is. **Always request landscape numbers.** In a portrait UI, a request such as 1440×1920 would match no native preset and fall into the resize path.

Recommended request:

```js
{ deviceId: { exact: wideId },
  width: { exact: 1920 }, height: { exact: 1440 },   // or 4032×3024 for max quality
  frameRate: { ideal: 30 } }
```

- On `OverconstrainedError`, retry with `ideal`. Since Safari 27.0 this error is a real `DOMException` ([RN 27][rn-27]).
- `aspectRatio` works but adds nothing here.
- `resizeMode` is not supported ([BCD]).

**Frame rate.** WebKit sets both `activeVideoMinFrameDuration` and `activeVideoMaxFrameDuration` to 1/fps ([WK][wk-avvcs]):

- The frame rate is constant, so auto-exposure cannot use exposures longer than 1/fps. In low light it raises ISO, which adds noise.
- 30 fps is the right request for this app.

**Switching formats live.** `applyConstraints()` can change format on a running track. Expect:

- a reconfiguration stall;
- the first 4 frames after a (re)start dropped (`framesToDropWhenStarting = 4`, [WK][wk-avvcs-h]);
- a `configurationchange` event, which is held until unmute since 27.0 ([RN 27][rn-27]).

On 26.x, never overlap `applyConstraints()` with `takePhoto()`. Safari 27.0 added queuing for this ([RN 27][rn-27]).

### 1.3 What `getSettings()` returns

Fields returned ([WK][wk-avvcs]):

- `width`, `height`. **These are swapped when frame rotation is ±90°**, for example `{width:1440, height:1920}` in a portrait UI.
- `frameRate`, `facingMode`, `deviceId`, `groupId` (both hashed per origin), `label`.
- `zoom` (normalized; see §2), `torch`, `whiteBalanceMode`, `backgroundBlur`, `powerEfficient`.

Not returned: focus, exposure, ISO or stabilization state.

### 1.4 Orientation

**How rotation is computed.** Rotation = sensor orientation − device orientation. The device orientation comes from WebKit's `OrientationNotifier`, which tracks the page/UI orientation. The `AVCaptureDeviceRotationCoordinator` path is disabled on iOS (that preference defaults to true only on Apple TV). ([WK][wk-avvcs], [WK prefs][wk-prefs]) With the iPhone's portrait orientation lock on, frames are therefore always tagged 90°.

**Buffers are not rotated by default.** Rotation stays metadata unless `setShouldApplyRotation()` has been called. That call sets a flag on the whole *source* (`m_isApplyingRotation`), not on a single sink. After it, `RealtimeMediaSource::videoFrameAvailable` rotates every frame for **all** observers of that source, MSTP included ([WK RealtimeMediaSource][wk-rms]).

- Callers on `main`: `MediaRecorderPrivateAVFImpl`, when the recorder wants rotation applied ([WK][wk-mravf]), and `RealtimeOutgoingVideoSource`, when WebRTC sinks ask for it.
- In the GPU-process path, the call is forwarded per remote source (`UserMediaCaptureManagerProxy::SetShouldApplyRotation`). A `track.clone()` gets its own remote source, so it should keep its own flag. That isolation is read from the code only (**UNVERIFIED** on device).
- **Rule:** use a dedicated clone for MSTP and never give that clone to `MediaRecorder` or `RTCPeerConnection`. Assert `codedWidth > codedHeight` on each frame, and start a new segment if the assertion fails.

What each API sees:

| API | Dimensions you get |
|---|---|
| `<video>`, `videoWidth`/`videoHeight`, rVFC `width`/`height` | Rotated (UI orientation) |
| **MSTP / `VideoFrame`** | `codedWidth`/`codedHeight` = raw `CVPixelBuffer` size, i.e. **sensor landscape, for example 1920×1440, in every UI orientation** ([WK MSTP][wk-mstp], [WK VideoFrameCV][wk-vfcv]). Safari has no `VideoFrame.rotation` ([BCD]). |
| `drawImage(videoFrame)`, `importExternalTexture`, `grabFrame` | Whether a hidden rotation is applied is **UNVERIFIED**. Test it on device (§8). |

**Transient rotation bug.** Safari 27.0 fixed "transient device rotation resulting in captured video frames having the wrong orientation" ([RN 27][rn-27], 180429147). On 26.x, if you use a rotated path, discard frames for about 0.5 s around each `screen.orientation` `change` event.

---

## 2. Device enumeration and choosing the fixed wide-angle lens

**Labels.** Labels are empty until the user grants camera permission. After that they are `AVCaptureDevice.localizedName`, **localized to the system language** ([WK `toCaptureDevice`][wk-avcdm]). Safari 26.0 fixed `enumerateDevices` reporting devices as available when permission had been denied ([RN 26][rn-26]).

**Exposed device types**, in WebKit's priority order ([WK `cameraCaptureDeviceTypes`][wk-avvcs]):

- `BuiltInTripleCamera`
- `BuiltInDualWideCamera`
- `BuiltInUltraWideCamera`
- `BuiltInDualCamera`
- `BuiltInWideAngleCamera`
- `BuiltInTelephotoCamera`
- `DeskViewCamera`
- `ExternalUnknown`

TrueDepth and LiDAR depth devices are **not** exposed.

**English labels.**

- Confirmed by the Scandit SDK's 33-language tables ([3P][scandit]): **"Back Camera"** (the physical wide lens), **"Back Ultra Wide Camera"**, **"Back Dual Wide Camera"**.
  - German: "Rückkamera", "Rückseitige Ultra-Weitwinkelkamera", "Rückseitige Dual-Weitwinkelkamera".
  - Japanese: "背面カメラ", and so on.
- Standard AVFoundation names, high confidence but **UNVERIFIED** here: "Back Triple Camera", "Back Dual Camera", "Back Telephoto Camera", "Front Camera".

**Expected device lists (UNVERIFIED per model):**

| iPhone class | Devices |
|---|---|
| Pro (11 Pro through 17 Pro) | Front, Back Triple, Back Dual Wide, Back Ultra Wide, Back Dual, **Back Camera**, Back Telephoto |
| Dual-lens non-Pro | Front, Back Dual Wide, Back Ultra Wide, **Back Camera** |
| Single-lens (SE, XR, 16e, Air) | Front, **Back Camera** |

**Order.** On iOS the default device (the front camera) is listed first. The rest follow discovery-session order ([WK][wk-avcdm]). **Do not rely on the order.**

**What `facingMode:"environment"` picks.** WebKit adds a fitness bonus by the priority list above, so a virtual Triple, Dual Wide or Dual camera wins whenever one exists ([WK `facingModeFitnessScoreAdjustment`][wk-avvcs]).

- **Virtual devices auto-switch lenses** based on zoom, light and focus position ([bug 250142][bug250142]).
- In the field: an iPhone 15 Pro on iOS 18.3.2 switched between wide, ultra-wide and telephoto mid-capture ([Apple forum 776460][forum776460]).
  - Constraints used: `facingMode:{exact:"environment"}`, `advanced:[{zoom:1}]`, and a `deviceId` whose value is not given.
  - Switching among three lenses implies that the Triple virtual device was in use. This report is **not** evidence that the physical "Back Camera" switches.

**Zoom normalization.** For Triple and DualWide cameras, WebKit multiplies zoom by 2 so that `zoom:1` gives the standard wide FOV (`min` 0.5 = ultra-wide). This was the fix for the iOS 16.4 "ultra-wide by default" regression (261877@main; [bug 253186][bug253186], [commit][wk-commit-261877]). The maximum is `min(videoMaxZoomFactor/scale, 10)`.

**Reliable way to get the fixed wide lens:**

1. Call `getUserMedia({video:{facingMode:{ideal:"environment"}}})` once to get permission, then `enumerateDevices()`.
2. Pick the `videoinput` whose label is in the localized **wide-label set** ("Back Camera" and translations, [scandit]).
3. If no label matches (unknown locale), use `InputDeviceInfo.getCapabilities()` (Safari 17+, [BCD]):
   - Keep `facingMode` ⊇ `environment` and **`zoom.min >= 1`**. That excludes Triple and DualWide, which report `zoom.min` 0.5 because WebKit applies a ×2 zoom scale.
   - **Correction:** `zoom.min >= 1` does **not** exclude the wide + telephoto **"Back Dual Camera"** (`BuiltInDualCamera`). Its zoom scale factor is 1, so it reports `zoom.min` = 1, the same as the physical wide camera ([WK `cameraZoomScaleFactor`][wk-avvcs], [WK VideoPreset.h][wk-vpreset]). It is present on Pro models and on wide + tele phones (for example the XS).
   - Exclude labels in the ultra-wide, telephoto, dual and triple sets. In an unknown locale the label filter does not work, and no exposed capability reliably tells Dual apart from the physical wide camera: `zoom.max` is clamped to 10 for both. A Dual camera at `zoom:1` is expected to stay on its wide lens (**UNVERIFIED**). If the choice is ambiguous, flag the session as "lens may switch" (step 5).
   - If more than one camera is left, take the one whose `focusDistance.min` is in the middle. The ultra-wide has the shortest minimum focus distance and the telephoto the longest. **UNVERIFIED heuristic.**
4. Re-open the camera with `deviceId:{exact:id}`. Assert that `track.label` matches and that `getSettings().zoom` is 1 when present.
5. **Last resort:** use the virtual device with `zoom:1`, flag the session as "lens may switch", and let SfM detect focal-length jumps (per-segment intrinsics).

---

## 3. Constraints and capabilities supported by Safari

| Constraint | Status | Notes |
|---|---|---|
| `zoom` | Settable | Physical wide camera: [1, min(maxZoom, 10)]. Virtual: [0.5, …]. Any value above 1 is a digital crop that changes intrinsics. **Keep it at 1.** |
| `torch` | Settable (when `hasTorch`) | Useful for dark close-range scenes. Adds moving specular highlights, so default it to off. |
| `focusMode` | No | Focus is AVFoundation's default (continuous AF is expected; **UNVERIFIED**). Focus breathing changes the focal length slightly; refine per segment. |
| `focusDistance` | Capability only | Use `focusDistance.min` (metres) to warn "too close". |
| `pointsOfInterest` | No | |
| `exposureMode`, `exposureCompensation`, `exposureTime`, `iso` | No | Auto-exposure always runs. Brightness varies between frames, so use exposure-invariant features and photometric normalization. |
| `whiteBalanceMode` | Settable | `"manual"` maps to `AVCaptureWhiteBalanceModeLocked`. **Lock it after AWB converges** for consistent texture colour. |
| `frameRate` | Settable | Fixed frame duration (see §1.2). |
| `resizeMode` | No | Request native sizes only. |
| `powerEfficient` | Settable (non-standard, iOS) | Leave unset or `false`. `true` prefers binned (often cropped) formats. |
| Video stabilization | Not applied | WebKit never sets `preferredVideoStabilizationMode`, and the AVF default is `.off` ([AVF][avf-stab]). Good: no per-frame warping. |
| Optical image stabilization (OIS) | Hardware; cannot be disabled from the web | Moves the principal point slightly. The magnitude is **UNVERIFIED** (a few px is expected). Let bundle adjustment refine `cx, cy`, or absorb it as noise. |
| Lens distortion correction | GDC defaults to `true` where supported ([AVF][avf-gdc]) | Applies to the ultra-wide and virtual devices that contain it; it "lessens the fisheye effect … upscales" ([AVF][avf-gdc-sup]). The wide camera most likely has no GDC and a small residual distortion (**UNVERIFIED**). Always estimate radial k1 and k2. |

---

## 4. Field of view and focal length in pixels

`videoFieldOfView` is the **horizontal FOV in the sensor's landscape orientation** ([AVF][avf-fov]). When the UI is portrait and you use a rotated path, it becomes the *vertical* FOV. Focal length in pixels is `f_px = W / (2·tan(HFOV/2))`.

**Measured values:**

| Device / camera (source) | Formats | HFOV | f_px at width W |
|---|---|---|---|
| iPhone 12 Pro Max, wide 26 mm ([zenn]) | 4:3 full (1920×1440 … 4032×3024) | 67.10° | 0.754·W → **1448 px @1920**, **3040 px @4032** |
| | 16:9 (1920×1080, 3840×2160) | 68.16° | 0.739·W → 1419 px @1920, 2838 px @3840 |
| | Low-res 4:3 (640×480, 1024×768) | 64.72° | 0.789·W → 505 px @640 |
| | Binned 1920×1440 / 1440×1080 | 63.90° / 59.31° | Hidden by dedup / cropped |
| iPhone X / 8, wide ([AVF archive][avf-archive]) | 4:3 / 16:9 | 58.90° / 60.98° | 0.886·W / 0.849·W |
| iPhone 8 Plus, telephoto ([AVF archive][avf-archive]) | 4:3 / 16:9 | 32.49° / 33.91° | — |

**Estimates for current devices (UNVERIFIED).** I derived these from Apple's 35 mm-equivalent focal lengths, using a 4:3 crop of the full-frame diagonal (34.6 mm wide). That method reproduces the 12 Pro Max measurement within 0.2°, but it was off by about 4° on the iPhone X.

| Camera | 35 mm-eq | HFOV (4:3) | VFOV (4:3) | f_px / W | f_px @1920 | f_px @4032 |
|---|---|---|---|---|---|---|
| Main, non-Pro (11–17, Air, 16e) | 26 mm | ≈67.3° | ≈53.1° | ≈0.751 | ≈1442 | ≈3029 |
| Main, Pro (14 Pro – 17 Pro) | 24 mm ([Apple specs][apple-17pro]) | ≈71.6° | ≈56.8° | ≈0.693 | ≈1331 | ≈2796 |
| Ultra-wide (all) | 13 mm, "120°" diagonal | ≈106° (before GDC) | ≈90° | ≈0.376 | ≈721 | ≈1514 |

**Cautions:**

- AVFoundation reported a bogus 39.3° for the 14 Pro Max wide camera on iOS 16.0 ([forum 714978][forum714978]). Treat every single value as a prior, not ground truth.
- **Recommended:** initialize `fx = fy = k·W`, `cx = W/2`, `cy = H/2`, `k1 = k2 = 0`, with a ±10% prior on `f`. Self-calibrate in bundle adjustment.
- Key the intrinsics by (camera label, format, plus model if you can infer it). `takePhoto()` JPEGs carry EXIF `FocalLength` and `FocalLengthIn35mmFilm` (WebKit uses `fileDataRepresentation`, [WK][wk-avvcs]). That EXIF content is **UNVERIFIED** but high confidence, and it makes a good per-device prior.

---

## 5. Frame access APIs

### 5.1 `requestVideoFrameCallback` (Safari 15.4)

For a camera `MediaStream`, WebKit fills these fields ([WK MediaPlayerPrivateMediaStreamAVFObjC][wk-mpms], [WK VideoFrameMetadata][wk-vfm-idl]):

| Field | Value |
|---|---|
| `captureTime` | `MonotonicTime::now()` when AVFoundation delivers the sample buffer to WebKit's capture delegate, i.e. **after ISP processing, not at exposure** ([WK][wk-avvcs]). |
| `presentationTime` | When the web-process player received the frame. |
| `expectedDisplayTime` | **Equal to `presentationTime`** (not a vsync prediction). |
| `width`, `height` | Rotated intrinsic size. |
| `presentedFrames` | Populated. |
| `mediaTime` | Always 0. |
| `processingDuration` | Normally absent. |
| `receiveTime`, `rtpTimestamp` | WebRTC remote streams only. |

All timestamps are coarsened to the performance timeline's reduced resolution. Safari 27 gives sub-millisecond `rAF` timestamps when the page is cross-origin isolated ([RN 27][rn-27]).

### 5.2 WebCodecs `VideoFrame` (Safari 16.4)

- `new VideoFrame(videoEl)` clones the *current playback frame*. It is IOSurface-backed, so there is no pixel copy ([WK][wk-wcvf]). `codedWidth` is the raw buffer size.
- `copyTo()` gives NV12 planes. A `format` conversion option exists in WebKit `main`, but BCD says unsupported: feature-detect it (**UNVERIFIED** which version ships it).
- Safari 27.0 fixed NV12 colour-range handling in the `VideoFrame` constructor and chroma offsets with `visibleRect` ([RN 27][rn-27]).
- `VideoFrame` is `Transferable` ([WK IDL][wk-wcvf-idl]).

### 5.3 `MediaStreamTrackProcessor` and `VideoTrackGenerator` (Safari 18.0)

**MSTP:**

- `Exposed=DedicatedWorker` only. It is **not** on Window, unlike Chrome. ([WK IDL][wk-mstp-idl], [BCD])
- `maxBufferSize` defaults to **1**. When the queue is full the oldest frame is dropped, so a slow consumer never builds latency ([WK][wk-mstp]).
- Frames are `"NV12"` from `kCVPixelFormatType_420YpCbCr8BiPlanarFullRange` on iOS ([WK RealtimeVideoUtilities][wk-rvu]).
- `timestamp` = the `CMSampleBuffer` output presentation timestamp in µs ([WK VideoFrameCV][wk-vfcv]). This is the capture clock. Whether it equals host time unchanged after crossing from the GPU process to the web process is **UNVERIFIED**; measure the offset as described in §7.
- Safari 26.4 fixed MSTP to honour `track.enabled = false` ([RN 26.4][rn-264]).

**`VideoTrackGenerator`:** Safari 18, worker only ([BCD]).

### 5.4 Transfers

| Object | Transferable? | Evidence |
|---|---|---|
| `MediaStreamTrack` | Yes, Window ↔ DedicatedWorker (`SameAgentCluster`). Safari 27.0 fixed wrong settings when a source changed during transfer. | [WK IDL][wk-mst-idl], [RN 27][rn-27] |
| `VideoFrame` | Yes | [WK IDL][wk-wcvf-idl] |
| `ReadableStream` | Yes, since 27.0 | [RN 27][rn-27] |

### 5.5 `ImageCapture` (Window only)

- **`takePhoto()` (18.4):**
  - Adds an `AVCapturePhotoOutput` on first use. Output is JPEG at quality 1.0 via `fileDataRepresentation` ([WK][wk-avvcs]).
  - **Photo size (corrected).** `maxPhotoSizeForFormat()` ([WK][wk-avvcs]) is *meant* to pick the smallest `supportedMaxPhotoDimensions` of the active video format that is ≥ the requested `imageWidth×imageHeight`. It does not do that:
    - It starts with the array's **first** entry.
    - It replaces that entry only with a candidate that is ≥ the request *and* has a smaller area.
    - AVFoundation lists the array in ascending order; for example, an iPhone 15 Pro prints `[4032×3024, 8064×6048]` ([forum 738418][forum738418]).
    - So the result is always the first, smallest entry, and `imageWidth`/`imageHeight` cannot increase it.
    - **Expect 4032×3024 from a 4:3 full-sensor format. 8064×6048 (48 MP) is unreachable from the web.** Whether some formats list a smaller first entry (for example equal to the video size) is **UNVERIFIED**. Check the JPEG size on device.
  - `getPhotoCapabilities()` ranges equal the track's width and height ranges. A request above `width.max` (for example 8064) is probably rejected (**UNVERIFIED**). Either way it would not be honoured, for the reason above.
  - **UNVERIFIED:** shutter sound on every call, capture latency (100–300 ms is my estimate), and whether ISP or multi-frame fusion warps geometry. Serialize calls; required on 26.x ([RN 27][rn-27]).
- **`grabFrame()` (26.0):** returns an `ImageBitmap`. Rotation behaviour is **UNVERIFIED**.

### 5.6 `createImageBitmap(video)` (Safari 15)

Each call does a GPU YUV→BGRA conversion and a new 4-byte-per-pixel allocation: about 11 MB at 1920×1440 and about 49 MB at 4032×3024. Timing is **UNVERIFIED** (an estimate of a few ms GPU plus allocation). Use it only as a fallback, and `close()` each bitmap immediately.

### 5.7 OffscreenCanvas, WebGPU and canvas limits

**OffscreenCanvas support:** 2D 16.4, WebGL/WebGL2 17, `"webgpu"` 26; `transferControlToOffscreen` 16.4; worker `requestAnimationFrame` 16.4 ([BCD]).

**WebGPU (Safari 26):**

- `importExternalTexture({source: VideoFrame})` gives zero-copy sampling of NV12 with automatic YUV→RGB conversion.
- `copyExternalImageToTexture({source: VideoFrame})` and `navigator.gpu` in workers are supported ([BCD]).

**Canvas area:**

| Version | Maximum area on iOS |
|---|---|
| Safari 18+ | **8192×8192 = 67,108,864 px** ([WK 276145@main][wk-canvas8k], [CanvasBase.cpp][wk-canvasbase]) |
| Earlier | 4096×4096 = 16.7 MP |

- The global "total canvas memory" limit was removed in Safari 17 ([WK 265628@main][wk-canvasmem]).
- Beyond the limit, context creation fails with the console message "Canvas area exceeds the maximum limit".
- WebContent process memory (jetsam) is now the real limit. The exact number is **UNVERIFIED**, so budget less than about 1 GB.

**Fingerprinting noise:**

- WebKit injects noise into canvas readback when `NoiseInjectionPolicy::Minimal` is active (Advanced Fingerprinting Protection), and for "known fingerprinting scripts" ([WK CanvasBase.cpp][wk-canvasbase]).
- Advanced Fingerprinting Protection is on by default in **Private Browsing**. Per Apple's description it adds noise to 2D-canvas and WebGL readback ([3P][privacyguides], [WK blog WWDC25][wk-wwdc25]).
- Whether camera pixels drawn into a canvas get noise in Private Browsing is **UNVERIFIED**, but the code path suggests yes.
- **Avoid `getImageData`/`readPixels` for image data.** Use `VideoFrame.copyTo` or WebGPU instead. At startup, run a known-pattern readback self-test.

---

## 6. Wake Lock, screen orientation, backgrounding

**Screen Wake Lock** — `navigator.wakeLock.request('screen')`:

- Safari 16.4 in the browser; works in Home Screen web apps only from **iOS 18.4** (webkit.org/b/254545; [BCD]).
- The lock is released when the page is hidden. Re-request it on `visibilitychange` → `visible`.
- Request it from the "Start capture" tap. Whether WebKit requires a user gesture is **UNVERIFIED**.

**`screen.orientation`:**

- `type`, `angle` and the `change` event work (16.4).
- **`lock()`/`unlock()` are not available.** The `ScreenOrientationLockingAPIEnabled` setting is "testable" and defaults to false ([WK prefs][wk-prefs], [BCD]).
- The manifest `orientation` member is ignored ([BCD]).
- **Design the capture UI and pipeline for both orientations.**

**Backgrounding:**

- On iOS, `InterruptVideoOnPageVisibilityChangeEnabled` is true ([WK prefs][wk-prefs]). A tab switch, app switch or screen lock interrupts the AVCaptureSession: the track fires `mute`, then `unmute` when it becomes visible again. The track normally does *not* end.
- **Known edge cases:**
  - A very quick background/foreground could leave the track muted, or skip `visibilitychange` ([bug 230538][bug230538], fixed on iOS 15; comments on [bug 232599][bug232599]).
  - `history.pushState` muted the track on iOS 17.4.1 ([bug 272704][bug272704]).
  - Audio tracks *end* after about 30 s in the background ([bug 204681][bug204681]). **Request no audio.**
- **Recovery:** on `visible`, if the track is still `muted` after about 1 s, or `readyState === "ended"`, stop it and call `getUserMedia` again with the same `deviceId`. Either way, **start a new SfM segment and relocalize**.

---

## 7. Recommended capture strategy for this app

1. **Gate on features.** Required: HTTPS, `MediaStreamTrackProcessor` in a worker (Safari 18+), WebGPU (Safari 26+).
   - Fallback ladder:
     - **A** (preferred): worker MSTP.
     - **B**: main-thread rVFC → `new VideoFrame(video)` → `postMessage(frame, [frame])`.
     - **C**: `createImageBitmap(video)` → transfer → worker `copyExternalImageToTexture`.
     - **D** (avoid): canvas `drawImage` + `getImageData`, because of Private Browsing noise and copy cost.
2. **Get permission and choose the camera** as in §2: physical **"Back Camera"**, never Triple/DualWide/Dual.
3. **Open the camera** with `deviceId:{exact}`, landscape-native sizes and `frameRate:{ideal:30}`, audio off.
   - **Standard mode:** 1920×1440. Full 4:3 FOV, about 4 MB of NV12 per frame, cheap enough to track every frame.
   - **Quality mode:** 4032×3024. Same FOV on measured devices, so intrinsics scale by exactly 2.1 (**UNVERIFIED on 48 MP sensors**). Process fewer frames; MSTP `maxBufferSize: 1` drops the rest.
   - Optional: `takePhoto()` keyframes, as a "still" mode, after the shutter-sound and latency checks in §8. Expect 4032×3024 with EXIF; 48 MP is not reachable (§5.5).
4. **Validate after open:**
   - `label` is the wide label.
   - `{width,height}` ∈ {requested, swapped}.
   - `zoom` is 1 (if present), `backgroundBlur` is false.
   - Record `focusDistance.min` (to warn the user when too close) and `getCapabilities()` (for logs and model inference).
5. **Lock white balance.** After about 1 s, if `capabilities.whiteBalanceMode` includes `"manual"`, call `applyConstraints({whiteBalanceMode:"manual"})`. Leave torch off by default; offer it as a toggle for close-range dark scenes. Never zoom.
6. **Run the pipeline:**
   - Keep the original track in `<video autoplay playsinline muted>` for preview.
   - Transfer `track.clone()` to the worker. Give both the same constraints; different constraints make WebKit add resize adaptors ([WK][wk-rvcs]).
   - Never pass the worker's clone to `MediaRecorder` or `RTCPeerConnection`. Either can switch its source to pixel rotation (§1.4). If you need a recording, make another clone.
   - In the worker: `new MediaStreamTrackProcessor({track, maxBufferSize: 2})` → for each `VideoFrame`:
     - either `importExternalTexture` for WebGPU compute (grayscale, pyramid, features), or `copyTo` of the Y plane;
     - assert `frame.codedWidth > frame.codedHeight` (sensor landscape). If it fails, the frames are being pixel-rotated: swap the intrinsics or start a new segment;
     - `frame.close()` **immediately**. The capture buffer pool is small; holding frames stalls capture (**UNVERIFIED** pool size).
   - Choose keyframes by sharpness (Laplacian variance on Y), parallax and IMU motion.
7. **Timestamps:**
   - Frame time comes from `frame.timestamp` (µs).
   - Estimate the mapping to `performance.now()` as the running minimum of `(performance.now()·1000 − frame.timestamp)`. Re-estimate after each interruption.
   - Refine camera–IMU time offset by correlating gyro rate (`devicemotion` `rotationRate`, stamped with `event.timeStamp`) with visual rotation.
   - Do not use rVFC `captureTime`: it is post-ISP arrival time.
   - Serve the page cross-origin isolated (COOP/COEP) for better timer precision and `SharedArrayBuffer`.
8. **Orientation:**
   - Run SfM on sensor-landscape buffers. MSTP frames are already landscape, so intrinsics never change with UI rotation.
   - Record `screen.orientation.angle` and the IMU gravity vector per frame to recover "up".
   - On the fallback paths (rotated frames), un-rotate, or swap `fx↔fy` and `cx↔cy`.
9. **Lifecycle:**
   - Request the wake lock on start and on `visible`.
   - On `mute`: mark a segment break. On `unmute` or re-acquire: relocalize.
   - On `pagehide`: flush state to IndexedDB/OPFS.
10. **Range of scenes:**
    - Up close: warn below `focusDistance.min` (≈0.15–0.2 m on the main camera; **UNVERIFIED** per model).
    - Far (backyard, houses): same wide camera and format. Optionally offer an "ultra-wide" mode on the physical "Back Ultra Wide Camera", with separate intrinsics and a stronger distortion model. GDC is on and cannot be disabled.

Snippet: choose the camera and hand it to a worker.

```js
const WIDE = new Set(["Back Camera","Rückkamera","Caméra arrière","Cámara trasera","背面カメラ","后置相机" /* …full list: Scandit */]);
let s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } }, audio: false });
s.getTracks().forEach(t => t.stop());
const cams = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === "videoinput");
let wide = cams.find(d => WIDE.has(d.label)) ?? cams.find(d => {
  const c = d.getCapabilities?.(); return c?.facingMode?.includes("environment") && (c.zoom?.min ?? 1) >= 1 && // excludes Triple/DualWide only;
    !/ultra|tele|dual|triple/i.test(d.label);                                                                   // "Back Dual Camera" (wide+tele) also has zoom.min 1
});
s = await navigator.mediaDevices.getUserMedia({ audio: false, video: {
  deviceId: wide ? { exact: wide.deviceId } : undefined, facingMode: wide ? undefined : "environment",
  width: { ideal: 1920 }, height: { ideal: 1440 }, frameRate: { ideal: 30 } } });
const track = s.getVideoTracks()[0];
const st = track.getSettings();                      // expect 1920×1440 or 1440×1920 (portrait)
const t2 = track.clone(); worker.postMessage({ track: t2 }, [t2]);
// worker: const r = new MediaStreamTrackProcessor({ track, maxBufferSize: 2 }).readable.getReader();
//         for (;;) { const { value: f } = await r.read(); /* f.format==="NV12", f.codedWidth=1920; assert codedWidth>codedHeight */ f.close(); }
```

---

## 8. On-device validation checklist (resolves the UNVERIFIED items)

Build a `/diag` page that logs the following on each target iPhone, on both iOS 26.x and 27.x:

1. `enumerateDevices()` labels and `getCapabilities()` per device (zoom, focusDistance, width/height max, frameRate).
2. For each candidate size, the result of `exact` constraints, `getSettings()` in portrait and in landscape, and MSTP `codedWidth×codedHeight`, `format`, `colorSpace.fullRange`.
3. Whether `drawImage(videoFrame)`, `importExternalTexture(videoFrame)`, `grabFrame()` and `createImageBitmap(video)` come out rotated in a portrait UI.
4. `frame.timestamp/1000 − performance.now()` (expect a constant offset with a few ms of jitter), and the frame interval at 30 fps.
5. A checkerboard calibration of fx, fy, cx, cy, k1, k2 at 1920×1440 and 4032×3024 on the "Back Camera" of a 24 mm Pro and a 26 mm non-Pro phone. Compare with the §4 priors.
6. Virtual camera with `zoom:1` approaching an object at 10 cm: detect the macro lens switch (FOV jump). This confirms the need for the physical camera.
7. `takePhoto()`: latency, shutter sound, EXIF focal fields, size versus the active format, the size with and without `{imageWidth: 8064, imageHeight: 6048}` (expected: no change, §5.5), and whether the preview stalls.
8. Private Browsing: a known-pattern canvas `getImageData` readback for noise.
9. Background/foreground cycles (fast and slow), screen lock, a phone call, and Control Center: `mute`/`unmute`/`ended` sequence and wake-lock re-acquisition.
10. Memory: sustained 4032×3024 MSTP plus WebGPU for 10 minutes without jetsam.

---

## Sources

- [rn-index]: https://developer.apple.com/documentation/safari-release-notes — Safari Release Notes index (versions and dates)
- [rn-26]: https://developer.apple.com/documentation/safari-release-notes/safari-26-release-notes — Safari 26.0 (`ImageCapture.grabFrame`, WebGPU, enumerateDevices fixes)
- [rn-264]: https://developer.apple.com/documentation/safari-release-notes/safari-26_4-release-notes — Safari 26.4 (MSTP `track.enabled` fix)
- [rn-27]: https://developer.apple.com/documentation/safari-release-notes/safari-27-release-notes — Safari 27.0 (rotation fix 180429147, ImageCapture queueing 174950018, VideoFrame fixes, track-transfer fix 172657570, OverconstrainedError, ReadableStream transfer)
- Safari 26.1, 26.2, 26.3, 26.5, 26.6 and 27.2-beta release notes were checked; they contain no further camera-capture features beyond minor WebRTC fixes.
- [wk-18]: https://webkit.org/blog/15865/webkit-features-in-safari-18-0/ — "MediaStreamTrack processing in a dedicated worker"
- [wk-wwdc25]: https://webkit.org/blog/16993/news-from-wwdc25-web-technology-coming-this-fall-in-safari-26-beta/ — fingerprinting protections for known scripts, WebGPU
- [wk-avvcs]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/mediastream/cocoa/AVVideoCaptureSource.mm — presets, device priority, zoom scale, settings and capabilities, rotation, takePhoto
- [wk-avvcs-h]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/mediastream/cocoa/AVVideoCaptureSource.h — `framesToDropWhenStarting = 4`
- [wk-avcdm]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/mediastream/cocoa/AVCaptureDeviceManager.mm — enumeration, labels, default device
- [wk-rvcs]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/mediastream/RealtimeVideoCaptureSource.cpp — preset matching and resize
- [wk-rms]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/mediastream/RealtimeMediaSource.cpp — rotation applied only on request; the flag is per source, not per sink
- [wk-mravf]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/mediarecorder/MediaRecorderPrivateAVFImpl.cpp — MediaRecorder calls `source->setShouldApplyRotation()`
- [wk-vpreset]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/mediastream/VideoPreset.h — default `minZoom` = 1
- [wk-rvu]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/mediastream/cocoa/RealtimeVideoUtilities.h — NV12 full range on iOS
- [wk-mc]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/mediastream/MediaConstraints.cpp — default 640×480 at 30 fps
- [wk-mtc]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/Modules/mediastream/MediaTrackConstraints.idl and MediaTrackCapabilities.idl — supported constraint list
- [wk-mst-idl]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/Modules/mediastream/MediaStreamTrack.idl — Transferable, worker-exposed
- [wk-mstp]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/Modules/mediastream/MediaStreamTrackProcessor.cpp — buffer size, timestamp, coded size
- [wk-mstp-idl]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/Modules/mediastream/MediaStreamTrackProcessor.idl — `Exposed=DedicatedWorker`
- [wk-vfcv]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/graphics/cv/VideoFrameCV.mm — PTS from CMSampleBuffer, presentationSize = buffer size
- [wk-wcvf]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/Modules/webcodecs/WebCodecsVideoFrame.cpp — `new VideoFrame(video)`
- [wk-wcvf-idl]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/Modules/webcodecs/WebCodecsVideoFrame.idl — Serializable/Transferable, copyTo options
- [wk-mpms]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/graphics/avfoundation/objc/MediaPlayerPrivateMediaStreamAVFObjC.mm — rVFC metadata for MediaStream
- [wk-vfm-idl]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/html/VideoFrameMetadata.idl
- [wk-prefs]: https://github.com/WebKit/WebKit/blob/main/Source/WTF/Scripts/Preferences/UnifiedWebPreferences.yaml — InterruptVideoOnPageVisibilityChange (iOS true), ScreenOrientationLockingAPIEnabled (false), UseAVCaptureDeviceRotationCoordinatorAPI (Apple TV only), MediaStreamTrackProcessing/ImageCapture (Cocoa true)
- [wk-canvasbase]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/html/CanvasBase.cpp — iOS 8192² area, noise injection policy
- [wk-canvas8k]: https://www.mail-archive.com/webkit-changes@lists.webkit.org/msg211769.html — 276145@main "[iOS] Increase the limit on the canvas size to 8192x8192" (2024-03-15)
- [wk-canvasmem]: https://github.com/WebKit/WebKit/commit/6bd11f3792f05b4e58e5647bf173212879fa62cc — 265628@main, total canvas memory limit removed
- [wk-commit-261877]: https://github.com/WebKit/WebKit/commit/1ed9893d37379f00ccdfb1da94c568bf4be79ef6 — zoom ×2 for Triple/DualWide
- [bug250142]: https://bugs.webkit.org/show_bug.cgi?id=250142 — support for AVCapture virtual cameras
- [bug253186]: https://bugs.webkit.org/show_bug.cgi?id=253186 — iOS 16.4 selects ultra-wide for environment
- [bug253646]: https://bugs.webkit.org/show_bug.cgi?id=253646 — expose zoom capability (261679@main)
- [bug248054]: https://bugs.webkit.org/show_bug.cgi?id=248054 — AVCaptureDeviceDiscoverySession enumeration (all physical cameras)
- [bug243075]: https://bugs.webkit.org/show_bug.cgi?id=243075, [bug280970]: https://bugs.webkit.org/show_bug.cgi?id=280970 — torch
- [bug204681]: https://bugs.webkit.org/show_bug.cgi?id=204681, [bug230538]: https://bugs.webkit.org/show_bug.cgi?id=230538, [bug232599]: https://bugs.webkit.org/show_bug.cgi?id=232599, [bug272704]: https://bugs.webkit.org/show_bug.cgi?id=272704 — background and mute behaviour
- [BCD]: https://github.com/mdn/browser-compat-data — api/MediaStreamTrackProcessor, VideoTrackGenerator, ImageCapture, VideoFrame, HTMLVideoElement, OffscreenCanvas, MediaStreamTrack, MediaDevices, InputDeviceInfo, WakeLock, ScreenOrientation, GPUDevice, GPUQueue, WorkerNavigator, XRSystem, createImageBitmap, manifests/webapp/orientation
- [avf-stab]: https://developer.apple.com/documentation/avfoundation/avcaptureconnection/preferredvideostabilizationmode — default `.off`
- [avf-gdc]: https://developer.apple.com/documentation/avfoundation/avcapturedevice/isgeometricdistortioncorrectionenabled — default true when supported
- [avf-gdc-sup]: https://developer.apple.com/documentation/avfoundation/avcapturedevice/isgeometricdistortioncorrectionsupported
- [avf-fov]: https://developer.apple.com/documentation/avfoundation/avcapturedevice/format/geometricdistortioncorrectedvideofieldofview (and `videoFieldOfView`)
- [avf-archive]: https://developer.apple.com/library/archive/documentation/DeviceInformation/Reference/iOSDeviceCompatibility/Cameras/Cameras.html — iPhone X/8 format tables with FOV
- [zenn]: https://zenn.dev/yorifuji/scraps/bc52d2e4601de0 — iPhone 12 Pro Max wide-camera format dump with FOV
- [forum776460]: https://developer.apple.com/forums/thread/776460 — lens switching in getUserMedia (iPhone 15 Pro, iOS 18.3.2)
- [forum738418]: https://developer.apple.com/forums/thread/738418 — 15 Pro supportedMaxPhotoDimensions 4032×3024 and 8064×6048
- [forum714978]: https://developer.apple.com/forums/thread/714978 — wrong videoFieldOfView on 14 Pro Max (iOS 16.0)
- [scandit]: https://unpkg.com/scandit-sdk@5.15.0/build/main/lib/cameraAccess.js — localized iOS camera label tables
- [privacyguides]: https://discuss.privacyguides.net/t/questions-about-safari-s-fingerprinting-protection/20513 — quotes Apple's canvas/WebGL readback noise description
- [apple-17pro]: https://www.apple.com/ph/iphone-17-pro/specs — 24 mm main camera

[rn-index]: https://developer.apple.com/documentation/safari-release-notes
[rn-26]: https://developer.apple.com/documentation/safari-release-notes/safari-26-release-notes
[rn-264]: https://developer.apple.com/documentation/safari-release-notes/safari-26_4-release-notes
[rn-27]: https://developer.apple.com/documentation/safari-release-notes/safari-27-release-notes
[wk-18]: https://webkit.org/blog/15865/webkit-features-in-safari-18-0/
[wk-wwdc25]: https://webkit.org/blog/16993/news-from-wwdc25-web-technology-coming-this-fall-in-safari-26-beta/
[wk-avvcs]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/mediastream/cocoa/AVVideoCaptureSource.mm
[wk-avvcs-h]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/mediastream/cocoa/AVVideoCaptureSource.h
[wk-avcdm]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/mediastream/cocoa/AVCaptureDeviceManager.mm
[wk-rvcs]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/mediastream/RealtimeVideoCaptureSource.cpp
[wk-rms]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/mediastream/RealtimeMediaSource.cpp
[wk-mravf]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/mediarecorder/MediaRecorderPrivateAVFImpl.cpp
[wk-vpreset]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/mediastream/VideoPreset.h
[wk-rvu]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/mediastream/cocoa/RealtimeVideoUtilities.h
[wk-mc]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/mediastream/MediaConstraints.cpp
[wk-mtc]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/Modules/mediastream/MediaTrackConstraints.idl
[wk-mst-idl]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/Modules/mediastream/MediaStreamTrack.idl
[wk-mstp]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/Modules/mediastream/MediaStreamTrackProcessor.cpp
[wk-mstp-idl]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/Modules/mediastream/MediaStreamTrackProcessor.idl
[wk-vfcv]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/graphics/cv/VideoFrameCV.mm
[wk-wcvf]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/Modules/webcodecs/WebCodecsVideoFrame.cpp
[wk-wcvf-idl]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/Modules/webcodecs/WebCodecsVideoFrame.idl
[wk-mpms]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/graphics/avfoundation/objc/MediaPlayerPrivateMediaStreamAVFObjC.mm
[wk-vfm-idl]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/html/VideoFrameMetadata.idl
[wk-prefs]: https://github.com/WebKit/WebKit/blob/main/Source/WTF/Scripts/Preferences/UnifiedWebPreferences.yaml
[wk-canvasbase]: https://github.com/WebKit/WebKit/blob/main/Source/WebCore/html/CanvasBase.cpp
[wk-canvas8k]: https://www.mail-archive.com/webkit-changes@lists.webkit.org/msg211769.html
[wk-canvasmem]: https://github.com/WebKit/WebKit/commit/6bd11f3792f05b4e58e5647bf173212879fa62cc
[wk-commit-261877]: https://github.com/WebKit/WebKit/commit/1ed9893d37379f00ccdfb1da94c568bf4be79ef6
[bug250142]: https://bugs.webkit.org/show_bug.cgi?id=250142
[bug253186]: https://bugs.webkit.org/show_bug.cgi?id=253186
[bug253646]: https://bugs.webkit.org/show_bug.cgi?id=253646
[bug243075]: https://bugs.webkit.org/show_bug.cgi?id=243075
[bug280970]: https://bugs.webkit.org/show_bug.cgi?id=280970
[bug204681]: https://bugs.webkit.org/show_bug.cgi?id=204681
[bug230538]: https://bugs.webkit.org/show_bug.cgi?id=230538
[bug232599]: https://bugs.webkit.org/show_bug.cgi?id=232599
[bug272704]: https://bugs.webkit.org/show_bug.cgi?id=272704
[BCD]: https://github.com/mdn/browser-compat-data
[avf-stab]: https://developer.apple.com/documentation/avfoundation/avcaptureconnection/preferredvideostabilizationmode
[avf-gdc]: https://developer.apple.com/documentation/avfoundation/avcapturedevice/isgeometricdistortioncorrectionenabled
[avf-gdc-sup]: https://developer.apple.com/documentation/avfoundation/avcapturedevice/isgeometricdistortioncorrectionsupported
[avf-fov]: https://developer.apple.com/documentation/avfoundation/avcapturedevice/format/geometricdistortioncorrectedvideofieldofview
[avf-archive]: https://developer.apple.com/library/archive/documentation/DeviceInformation/Reference/iOSDeviceCompatibility/Cameras/Cameras.html
[zenn]: https://zenn.dev/yorifuji/scraps/bc52d2e4601de0
[forum776460]: https://developer.apple.com/forums/thread/776460
[forum738418]: https://developer.apple.com/forums/thread/738418
[forum714978]: https://developer.apple.com/forums/thread/714978
[scandit]: https://unpkg.com/scandit-sdk@5.15.0/build/main/lib/cameraAccess.js
[privacyguides]: https://discuss.privacyguides.net/t/questions-about-safari-s-fingerprinting-protection/20513
[apple-17pro]: https://www.apple.com/ph/iphone-17-pro/specs

---

## Review notes

Reviewed on 2026-10-16 by a second agent. Sources: WebKit `main` (raw files fetched that day), the Apple Safari release-notes JSON (26.0, 26.4, 26.6, 27.0, 27.2 beta, 18.0, 18.4), MDN BCD `main`, AVFoundation documentation JSON, and the cited forum and zenn pages.

**Verified, no change needed:**

- **Format presets.** `generatePresets` takes one preset per size; the first format of each size wins. It skips only packed-Bayer ProRes RAW and applies no resolution cap. The zenn dump confirms the 12 Pro Max order: the first 1920×1440 format is 420v, 1–30 fps, 67.096°, not binned.
- **Camera choice and zoom.**
  - `cameraCaptureDeviceTypes` order is Triple > DualWide > UltraWide > Dual > Wide > Tele.
  - `facingModeFitnessScoreAdjustment` favours virtual back cameras.
  - Zoom scale ×2 applies to Triple and DualWide only; max zoom is `min(videoMaxZoomFactor/scale, 10)`.
  - Bug 253186 was fixed by 261877@main.
- **Capture settings.**
  - Fixed frame duration: min and max are both 1/fps.
  - `framesToDropWhenStarting = 4`.
  - Default constraints are ideal 640×480 at 30 fps.
  - rVFC `captureTime` is `MonotonicTime::now()` at the delegate, and `expectedDisplayTime` equals `presentationTime`.
- **Constraints.** `MediaTrackConstraints` and `MediaTrackCapabilities` IDL: focus, exposure, ISO and pointsOfInterest are FIXMEs. `focusDistance` is a capability only, as `minimumFocusDistance/1000` m. There is no `resizeMode`.
- **MSTP and VideoFrame.**
  - MSTP is `Exposed=DedicatedWorker`, with `maxBufferSize` defaulting to 1 and the oldest frame dropped.
  - `codedWidth` is the `CVPixelBuffer` width, and `timestamp` is the sample-buffer PTS.
  - iOS uses `420YpCbCr8BiPlanarFullRange`.
  - `VideoFrame` has no `rotation`, `flip` or `metadata` (WebKit IDL and BCD). BCD lists `copyTo` `format` as unsupported.
- **Release notes, all items confirmed verbatim:**
  - Safari 27.0: 180429147, 174950018, 172657570, 180728516, 180728609, 169950592, 170299037, 180202939, 176967366.
  - Safari 26.0: `grabFrame` 148425176, `enumerateDevices` 147313922.
  - Safari 26.4: MSTP `track.enabled` 165199900.
  - Safari 18.0: MSTP in a dedicated worker. Safari 18.4: Wake Lock in Home Screen web apps.
  - Release dates: 26.6 on 2026-07-27, 27.0 on 2026-09-14, 27.2 beta on 2026-10-13.
- **BCD versions:** ImageCapture 18.4, `grabFrame` 26, MSTP and VideoTrackGenerator 18, `InputDeviceInfo.getCapabilities` 17, rVFC 15.4, `WakeLock` 18.4 (partial from 16.4). `ScreenOrientation.lock` and `XRSystem` are unsupported.
- **Preferences.** `InterruptVideoOnPageVisibilityChangeEnabled` is true on iOS. `ScreenOrientationLockingAPIEnabled` is false. `UseAVCaptureDeviceRotationCoordinatorAPI` is on only on Apple TV.
- **Canvas.** The iOS canvas area limit is 8192² in `CanvasBase.cpp`.
- **AVFoundation defaults.** `preferredVideoStabilizationMode` defaults to `.off`. GDC is enabled by default where supported.

**Corrected:**

1. **`takePhoto()` size (§5.5, summary table, §7, §8).** The report said WebKit picks the smallest `supportedMaxPhotoDimensions` entry that is ≥ the request. The loop actually seeds with the first entry and accepts only smaller areas. With AVFoundation's ascending order, the request is ignored and 48 MP (8064×6048) cannot be reached.
2. **`zoom.min >= 1` heuristic (§2, snippet).** It does not exclude the wide + tele "Back Dual Camera": its zoom scale is 1, so `minZoom` defaults to 1.
3. **"Buffers are never rotated" (key conclusion 4, §1.4, §7).** `setShouldApplyRotation()` is a per-*source* flag that rotates frames for all observers, MSTP included. `MediaRecorder` and the WebRTC outgoing source set it. Added the rule to keep a dedicated MSTP clone and to assert `codedWidth > codedHeight` on each frame.
4. **Forum 776460 (key conclusion 1, §2).** The post never names its `deviceId`, and three-lens switching implies the Triple device. It is not evidence that the physical camera switches. That the physical camera does *not* switch is still unconfirmed on device, so it is now marked **UNVERIFIED**.

**Downgraded to UNVERIFIED or qualified:**

- That 1920×1440 is full-FOV and non-binned on 48 MP-sensor iPhones. Only a 12 Pro Max dump exists.
- `MediaStreamTrack` transfer "since 18.0". BCD has no entry; the version is inferred from MSTP being worker-only in 18.0.
