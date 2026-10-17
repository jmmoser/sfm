# Motion, orientation and location sensors in iOS Safari (iOS 26 / iOS 27) for visual-inertial SfM

Status: researched 2026-10-16. Scope: iOS/iPadOS Safari 26.0 through 27.0 (shipping), plus WebKit `main` (Oct 2026). The same engine runs in Home Screen web apps and in third-party iOS browsers, which use WKWebView.

**How claims are tagged** (same scheme as `ios-webgpu.md`)

- **[V]** Verified in a primary or authoritative source: a W3C spec, Apple developer docs, WebKit Bugzilla, the WebKit blog, or MDN browser-compat-data (BCD).
- **[S]** Read directly from WebKit source. The branches are `safari-7622-branch` (Safari 26.0–26.1), `safari-7625-branch` (Safari 27.0) and `main`. The sensor code is **identical on all three** and unchanged since at least June 2020 (commit `702afb7`). This is strong evidence of what the shipping binary does, but it is still not a device measurement.
- **[U]** UNVERIFIED. Each one comes with a best estimate and a safe fallback.

---

## 0. Summary table

| Question | Answer on iOS Safari 26/27 | Tag |
|---|---|---|
| Data source | CoreMotion `CMDeviceMotion` (sensor fusion). Polled by an **NSTimer at 1/60 s** in the UI process, then sent by IPC to the web process. Only the **main thread** (`Exposed=Window`) receives the event. | [S] [1][2][4][5] |
| Event rate | **About 60 Hz, fixed, not configurable.** It was 20 Hz before 2015 (r185443). ProMotion does not change it. | [S] [2], [V] [10] |
| `interval` | **Constant `0.016666667`, in *seconds*.** The spec says milliseconds. It is the nominal value, not a measured one. | [S] [3], spec [14] |
| Value precision | **Full `double` precision, not quantised.** The spec now requires limiting precision to 0.1 m/s², 0.1 deg/s and 0.1°, but WebKit's iOS path applies no rounding anywhere. | [S] [1][3], spec [14] |
| `rotationRate` units | **deg/s.** WebKit applies `rad2deg` to CoreMotion's rad/s. The rate is bias-compensated ("unbiased"). | [S] [1], [V] [19] |
| `rotationRate` axes | **alpha = about device x, beta = about device y, gamma = about device z.** Right-hand rule. This matches the current spec, which was changed in 2017 to match the browsers. | [S] [1], [V] [14][17] |
| `accelerationIncludingGravity` sign | **Inverted relative to the W3C spec on every axis.** Face-up gives **z ≈ −9.81** on iOS but **+9.81** per the spec and Chrome/Android. Held upright in portrait gives y ≈ −9.81. **There has been no change across versions**: reported in 2014, and the code is unchanged from 2020 through 2026. | [S] [1], [V] [16][20] |
| `acceleration` (gravity removed) | **Provided.** It is `9.80665 × CMDeviceMotion.userAcceleration` (CoreMotion fusion). **It is also inverted relative to the spec** (iOS reports −kinematic acceleration). | [S] [1], [V] [16] |
| Permission | `DeviceMotionEvent.requestPermission()` / `DeviceOrientationEvent.requestPermission()` share **one grant per top-level origin**. A **user gesture is required to prompt** (`click`/`touchend`, not `touchstart`). Both interfaces are `[SecureContext]`, so the page must be HTTPS. WebKit itself caches the grant **in memory only**. Safari can also inject a stored per-site Ask/Grant/Deny policy at navigation time through the `_WKWebsiteDeviceOrientationAndMotionAccessPolicy` SPI, so **whether the grant survives a relaunch is [U]**. | [S] [5][6][7], [V] [11][12][35] |
| `event.timeStamp` | Milliseconds on the `performance.now()` timeline. It is the **creation time in the web process** (after IPC), **floored to 1 ms**, or to **20 µs when cross-origin isolated**. There is no sensor timestamp. | [S] [8] |
| Latency / jitter | Sample age is 0–16.7 ms (polled) plus IPC plus main-thread queueing. Expected latency is about 10–20 ms, with σ about 2–5 ms on an idle main thread. Events arrive in bursts when the main thread is blocked. | [U] |
| DeviceOrientation `alpha` | **Arbitrary reference.** CoreMotion uses `xArbitraryZVertical`, so yaw = 0 is wherever the device pointed when updates started. It resets when updates restart. The angles follow the spec's Z-X'-Y'': α∈[0,360), β∈[−180,180), γ∈[−90,90). | [S] [1], [V] [19][18] |
| `webkitCompassHeading` | `CLHeading.magneticHeading`: **magnetic** north (not true), measured clockwise, referenced to the device top in portrait. It reads **0 with accuracy −1 when unavailable**. `webkitCompassAccuracy` = `headingAccuracy` in degrees. | [S] [1], [V] [21] |
| `deviceorientationabsolute`, `event.absolute` | **Neither is supported.** The attribute is compiled out on iOS. | [S] [5], BCD [25] |
| Generic Sensor API (Accelerometer, Gyroscope, LinearAcceleration, AbsoluteOrientation, Gravity, Magnetometer) | **None.** WebKit's standards position is negative. | BCD [25], [V] [26] |
| Geolocation | `enableHighAccuracy:true` maps to `kCLLocationAccuracyBest`; `false` maps to `kCLLocationAccuracyHundredMeters`. Expect about 1 Hz updates. `altitude` is **MSL (EGM2008), not WGS84 ellipsoidal**. `heading` and `speed` are course over ground and ground speed, and are `null` when invalid. `timestamp` is the fix time in epoch milliseconds. Window-only. With Precise Location off, accuracy drops to **1–20 km**. | [S] [22], [V] [23] |
| Screen rotation | `window.orientation` ∈ {0, 90, −90, 180}, where 90 = device turned **counter-clockwise** (top edge pointing left). **iPhone** `screen.orientation.angle` = `(window.orientation+360)%360`. **On iPad (natural landscape) the angle uses a different mapping**, so prefer `window.orientation`. `screen.orientation.lock()` is unsupported. | [S] [27], [V] [28][25] |
| Camera frame vs device axes | Camera frames are rotated upright for the current **interface** orientation θ. `R_cam←dev(θ) = [[c,−s,0],[−s,−c,0],[0,0,−1]]`. This matches `docs/ARCHITECTURE.md` §2. | [S] [30], math §5 |

---

## 1. DeviceMotionEvent

### 1.1 Data path (why the numbers are what they are) [S]

```
CoreMotion CMMotionManager (UI process, deviceMotionUpdateInterval = 1/60 s, default ref frame xArbitraryZVertical)
   │  NSTimer every 1/60 s on the UI-process main run loop polls `[motionManager deviceMotion]` (latest sample)
   ▼
WebCoreMotionManager::sendMotionData   (WebCoreMotionManager.mm)
   acceleration            = 9.80665 · userAcceleration
   accelerationIncludingG  = 9.80665 · (userAcceleration + gravity)          ← no sign change
   rotationRate(α,β,γ)     = rad2deg(rotationRate.x, .y, .z)                ← α=x, β=y, γ=z
   orientation(α,β,γ)      = Euler Z-X'-Y'' from CMAttitude; heading = CLHeading.magneticHeading
   ▼ IPC (WebDeviceOrientationUpdateProviderProxy → WebDeviceOrientationUpdateProvider)
Web process main thread: DeviceMotionClientIOS::motionChanged → DeviceMotionData(…, interval = kMotionUpdateInterval)
   ▼ DeviceMotionEvent created now (Event::m_createTime) and dispatched on window
```

Sources: [1] `Source/WebCore/platform/ios/WebCoreMotionManager.mm`, [2] `WebCoreMotionManager.h` (`constexpr float kMotionUpdateInterval = 1.0f / 60.0f;`), [3] `DeviceMotionClientIOS.mm`, and [4] `Source/WebKit/UIProcess/ios/WebDeviceOrientationUpdateProviderProxy.mm`. The proxy forwards the values unchanged.

### 1.2 Permission flow

- **API [S][V].** `DeviceMotionEvent.requestPermission()` and `DeviceOrientationEvent.requestPermission()` both exist and both return `Promise<"granted"|"denied">` [5]. They have existed since Safari 13 (iOS 13) [12]. BCD lists iOS 14.5, which conflicts with bug 201676. Either way, feature-detect `typeof DeviceMotionEvent.requestPermission === "function"` instead.
- **One shared grant [S].** Both methods go through the same `DeviceOrientationAndMotionAccessController`, keyed by the **top document's origin** [6]. Granting one grants both. Since there is no `absolute` variant on iOS, no separate magnetometer prompt exists.
- **User gesture [S][V].** `mayPrompt = UserGestureIndicator::processingUserGesture(document)` [6]. If the state is still "prompt" and no gesture is in progress, the promise **rejects with `NotAllowedError`** ("requires a user gesture to prompt") [5]. Rules that follow from this:
  - Call it **synchronously at the top of a `click` or `touchend` handler, before any `await`**. Safari does not treat `touchstart` as a user gesture [35].
  - A gesture that is round-tripped through a Worker `postMessage` is not honoured (bug 269706, still open) [13].
  - Request motion permission **first** in the tap handler, then call `getUserMedia` and `geolocation`, which do not need the gesture.
- **Caching [S][V].** A decision is stored in two places:
  - per WebCore `Page` (`m_accessStatePerOrigin`);
  - per `WebsiteDataStore` in the UI process (`m_deviceOrientationPermissionDecisions`, an in-memory `HashMap`) [6].

  Within the same browsing session a later `requestPermission()` therefore resolves `"granted"` **without a gesture** (bug 197750, r245185) [11]. The UI-process cache is keyed by the main frame's origin. The WebCore cache is keyed by the requesting document's origin and lives on the `Page`, so a reload in the same tab may keep receiving events without a new call, while a page in a new `Page` (new tab, or a process swap) receives none until `requestPermission()` is called again. That later call resolves silently. **There is no on-disk persistence in WebKit itself.**
  - **Embedder override [S].** `WKWebpagePreferences._deviceOrientationAndMotionAccessPolicy` (`_WKWebsiteDeviceOrientationAndMotionAccessPolicyAsk/Grant/Deny`) lets Safari seed `DocumentLoader::deviceOrientationAndMotionAccessState()` on each navigation. Safari can therefore persist decisions in its own per-site settings without WebKit storing anything [39].
  - **[U]** It is not known whether Safari 26/27 or Home Screen web apps actually persist the decision across relaunches. Plan for a prompt on every launch; a silent `"granted"` is a bonus.
- **Fallback when there is no gesture.** Call `requestPermission()` on startup anyway. If it rejects with `NotAllowedError`, show a "Start capture" button and call it again from that button's handler.
- **Other gates [S].** The page must be a secure context. Cross-origin iframes need `Permissions-Policy`/`allow` for **`accelerometer` and `gyroscope`**, plus **`magnetometer`** for `deviceorientation` [7]. The API is disabled in the iOS Simulator, where all values are null [3].
- **Standalone / Home Screen web app [V]/[U].** Since iOS 26, every site added to the Home Screen opens as a web app by default [32]. It runs the same WebKit code path, so the same prompt and gesture rules apply. Re-prompting on each launch is expected because the cache is in memory only [U]. The Settings toggle "Motion & Orientation Access" from iOS 12.2 was replaced by the per-site prompt in iOS 13 [12].
- **Detecting state.** `navigator.permissions.query({name:"accelerometer"})` is **not usable**. `"accelerometer"`, `"gyroscope"` and `"magnetometer"` are valid `PermissionName` values, but WebKit's `Permissions::toPermissionName` does not map them, so `query()` rejects with `NotSupportedError` [S]. BCD also lists `permission_accelerometer`/`permission_gyroscope` as unsupported in Safari [25]. Calling `requestPermission()` is the only reliable probe. If a listener is added in the denied **or not-yet-requested** state, WebKit logs "No device motion events will be fired, reason: …" to the console and fires nothing [7]. In a cross-origin iframe that lacks `allow="accelerometer; gyroscope"`, `requestPermission()` **resolves `"denied"`** instead of rejecting [5].

### 1.3 Event rate and `interval`

- **Rate [S][V].** `kMotionUpdateInterval = 1/60` drives both CoreMotion's `deviceMotionUpdateInterval` and the NSTimer that polls it [1][2]. Bug 145814 raised it from 20 Hz to 60 Hz in June 2015. A request for 100 Hz, plus per-event sample batching, was discussed in that bug but never landed [10]. **Nothing exposes a way to change it.**
- **`interval` [S].** WebKit passes the constant `kMotionUpdateInterval` (a float) as `interval`, so `event.interval === 0.01666666753590107`. That value is **in seconds**. The spec requires **milliseconds** [14], and Chromium reports milliseconds [U: exact value not checked]. **Normalise with `if (iv > 0 && iv < 1) iv *= 1000`. Never use `interval` for integration**, because it is not measured. Use `timeStamp` differences.
- **Duplicates and drops [S]/[U].** The timer reads the *latest* CoreMotion sample, and two unsynchronised 60 Hz clocks beat against each other. An event can therefore repeat the previous sample (bit-identical values) or skip one. **De-duplicate exact repeats.** Measured frequency is [U]; an estimate is about 1–5 % of events.
- **Replayed first event [S].** `DeviceController::addDeviceEventListener` re-dispatches the **last cached sample** on a 0 s timer whenever a listener is added [9]. **Drop the first event after (re)subscribing.**
- **Pauses [S]/[U].**
  - Removing the last listener stops CoreMotion, and re-adding it restarts CoreMotion, which **resets the DeviceOrientation yaw reference** [1][9].
  - iOS suspends updates when the page is suspended (`DeviceMotionController::suspendUpdates`) [S].
  - The NSTimer is scheduled in the default run-loop mode, so it probably **does not fire while the UI process is tracking a scroll gesture** [U, inference from `scheduledTimerWithTimeInterval`]. During capture, use non-scrolling UI (`touch-action: none`, `overflow: hidden`).
  - **Keep one permanent listener.**

### 1.4 Units

| Field | iOS unit | Spec unit | Conversion |
|---|---|---|---|
| `rotationRate.{alpha,beta,gamma}` | deg/s (`rad2deg` of CoreMotion's rad/s) [S][V] | deg/s [14] | × π/180 → rad/s |
| `acceleration.{x,y,z}`, `accelerationIncludingGravity.{x,y,z}` | m/s², computed as `g × 9.80665` [S] | m/s² | none needed |
| `interval` | **s** (constant) [S] | ms | × 1000 if < 1 |
| `timeStamp` | ms on the `performance.now()` timeline [S] | ms | none needed |

**Precision [S].** The current spec tells implementations to "limit precision" to 0.1 m/s² for acceleration, 0.1 deg/s for rotation rate and 0.1° for orientation angles, as an anti-fingerprinting measure [14]. WebKit's iOS path (`WebCoreMotionManager`, `DeviceMotionClientIOS`, `DeviceMotionData`, `DeviceOrientationData`) contains no rounding on `main` or on the 26/27 branches, so iOS delivers full-precision CoreMotion values.
- **[U]** Whether Chromium quantises was not checked.
- Do not hard-code any assumption about quantisation, in either direction. If values are quantised, the exact-repeat de-duplication in §7 can drop genuine stationary samples. Enable it only when the engine is WebKit, or when the repeat rate is high and the gyro is not quantised.

Historical note: Chrome on Android **before m66** reported `rotationRate` in rad/s. Libraries such as cardboard-vr-display still special-case it [33]. iOS has always used deg/s.

### 1.5 Axis convention

- **Device frame [V].** For a phone, the device frame is "defined relative to the screen in its standard orientation, typically portrait": +x toward the right edge, +y toward the top edge, +z out of the screen. **It does not rotate with the UI** [14]. CoreMotion uses the same body frame and WebKit passes it through unchanged [S].
- **iPad [U].** CoreMotion's axes are those of the hardware held in portrait, which is presumably also the frame WebKit uses. The self-checks in §7 confirm this at run time.
- **rotationRate mapping [S][V].** `alpha = ω_x`, `beta = ω_y`, `gamma = ω_z` [1]. The current spec's getters say exactly this: "alpha … x axis rotation rate", "beta … y", "gamma … z" [14]. The 2017 spec PR changed the text to match the existing behaviour of Chrome, Firefox and Safari [17]. Older code and docs that call alpha the "z rate" are wrong.
- **Sign [V][U].** Right-hand rule: positive means counter-clockwise when looking down the positive axis [14]. CoreMotion follows the same convention. A device check of this is [U], and the gyro self-check in §7 covers it.

### 1.6 Sign of `acceleration` and `accelerationIncludingGravity` (the important one)

- **CoreMotion convention [V].** The accelerometer reports gravity as a vector pointing **down**:
  - stationary and upright in portrait: `(0, −1, 0)` g;
  - lying face-up: `(0, 0, −1)` g [20].

  `CMDeviceMotion.gravity` is the gravity vector, and the total acceleration is `gravity + userAcceleration` [19].
- **What WebKit does [S].** It multiplies by 9.80665 and changes no signs [1].
  - `accelerationIncludingGravity` = 9.80665·(gravity + userAccel). Face-up this reads **(0, 0, −9.81)**.
  - `acceleration` = 9.80665·userAccel.
- **What the spec says [V].** `accelerationIncludingGravity` is the *proper acceleration* (specific force):
  - face-up gives `{0, 0, +9.8}`;
  - free fall gives `acceleration.z = −9.8`;
  - the vehicle example (screen vertical, facing the rear, right-hand bend) gives **positive x** for both fields [14][15].
- **Observed [V].** Tibbett, W3C list, July 2014: "In iOS Safari this configuration returns {x: 0, y: 0, z: -9.81}". In the vehicle case, "iOS Safari records a negative x component for both acceleration and accelerationIncludingGravity", while Android browsers follow the spec [16]. Chrome engineers agreed that iOS is the inverted one [16].
- **Conclusion.** On iOS Safari 26/27:
  - `accelerationIncludingGravity_iOS = −f_spec` (specific force negated);
  - `acceleration_iOS = −a_spec` (kinematic acceleration negated);
  - all three axes are affected.

  No WebKit change has touched this: the source is identical in 2020, on safari-7622 (26.0), on safari-7625 (27.0) and on main [S]. **Never sniff the user agent.** Use the run-time sign check in §7, which also survives a future fix.
- **Which field to use for VIO.** Use `accelerationIncludingGravity`, which reconstructs CoreMotion's calibrated total acceleration because userAcceleration + gravity add back up. Pre-integrate it and estimate gravity yourself. `acceleration` depends on CoreMotion's gravity estimate, which lags during fast rotation. It is still useful for detecting stillness and as a cross-check.
- **Magnitude [U].** At rest, |accelerationIncludingGravity| ≈ 9.81 ± (accelerometer scale error plus local-g variation of about 0.3 %). WebKit always multiplies by 9.80665, whatever the local g.

### 1.7 `acceleration` (gravity removed) on iOS

**It is available whenever the device has a gyroscope,** which every iOS 26 device does [S]. It is `CMDeviceMotion.userAcceleration`, the output of CoreMotion's accelerometer+gyro fusion [1][19]. On a device without a gyro, WebKit would fall back to `CMAccelerometerData` and report `acceleration = {0,0,0}` (zeros, not null) and `rotationRate = null` [S]. That path does not occur on supported iPhones.

### 1.8 `timeStamp`, latency and jitter

- **Time base [S].** `Event::timeStampForBindings` returns `performance.relativeTimeFromTimeOriginInReducedResolution(m_createTime)` [8]. That is the **same clock as `performance.now()`** in the document. `m_createTime` is the moment the web process created the event after receiving IPC. **No CoreMotion sensor timestamp (`CMLogItem.timestamp`) reaches JavaScript** [1][3].
- **Resolution [S].** It is floored to **1 ms** by default (`static Seconds timePrecision { 1_ms }`). It becomes **20 µs** once the process is cross-origin isolated (`ScriptExecutionContext::setCrossOriginMode(Isolated)` → `Performance::allowHighPrecisionTime()` → `Seconds::highTimePrecision() = 20 µs`) [8]. Safari 27 also fixed sub-millisecond rAF timestamps under cross-origin isolation [31]. The app already needs COOP/COEP for Wasm threads, so it also gets these finer timestamps.
- **Latency budget [U].**
  - Sample age when the timer polls: uniform 0–16.7 ms (mean about 8 ms).
  - Run-loop jitter: about 1–2 ms.
  - IPC: under 1–2 ms.
  - Web-process main-thread queueing: 0 when idle, unbounded when busy.
  - Expected total offset is about 10–20 ms behind the physical sample, with σ about 2–5 ms on an idle main thread.
  - If the main thread stalls (say a 100 ms task), about 6 queued events are dispatched **back-to-back with nearly identical timestamps**. cardboard-vr-display also warns that iOS can produce non-monotonic or out-of-range `timeStamp` values [33].
- **Mitigations.**
  - Keep the main thread nearly idle. All SfM work runs in workers.
  - Forward samples to the worker through a SharedArrayBuffer ring buffer.
  - Reject `dt ≤ 1 ms` (bursts) and `dt > 100 ms` (gaps) and resample.
  - Estimate a camera–IMU time offset `t_d` in the VIO, as VINS-Mono does, starting from about −15 ms (IMU sample time ≈ `timeStamp − 15 ms`) with a ±50 ms search range [U].

---

## 2. DeviceOrientationEvent

- **Rate and coupling [S].** It is produced in the **same timer tick, from the same `CMDeviceMotion` sample**, as `devicemotion` (motion first, then orientation) [1]. Events therefore arrive in pairs at 60 Hz with consistent data, which is ideal for the cross-checks in §7.
- **Angle convention [S].** WebKit builds `R = Rz(yaw)·Rx(pitch)·Ry(roll)` from `CMAttitude` and extracts the spec's intrinsic Z-X'-Y'' angles [1]:
  - `alpha ∈ [0, 360)`: rotation about the world vertical, counter-clockwise looking down;
  - `beta ∈ [−180, 180)`: about device x;
  - `gamma ∈ [−90, 90)`: about device y.

  This matches the spec [14]. The older iOS ranges (β ±90, γ ±180) described in 2014 [18] no longer apply. The device "up" vector follows directly: `up_dev = (−cosβ·sinγ, sinβ, cosβ·cosγ)` (§7).
- **`alpha` reference [S][V].** `startDeviceMotionUpdates` is called without a reference frame, so CoreMotion uses its **default `xArbitraryZVertical`**. In that frame, "Core Motion sets the frame of reference to the device's initial orientation" and does not use the magnetometer [19]. So:
  - **`alpha` is relative to an arbitrary yaw**: in practice, wherever the device pointed when updates started (Tibbett, 2014: "zero-centered on the direction the page started listening") [18];
  - it **resets** whenever motion updates restart (last listener removed, page suspended);
  - it drifts slowly with gyro yaw drift. An estimate is roughly 1–10°/min [U].
- **`absolute` / `deviceorientationabsolute` [S][V].** On iOS the IDL compiles out `absolute` (`event.absolute === undefined`). `deviceorientationabsolute` is not among WebKit's event names [5]. BCD agrees on both [25].
- **`webkitCompassHeading` [S][V].** `CLHeading.magneticHeading`, from `CLLocationManager startUpdatingHeading` in the UI process [1].
  - **Magnetic north**, degrees, clockwise (0 = N, 90 = E) [21].
  - Referenced to the **top of the device in portrait**, because WebKit never sets `headingOrientation` [21].
  - Updates only after **≥ 1° of change** (default `headingFilter` = 1°) [21].
  - When no heading is available WebKit reports **0** and `webkitCompassAccuracy = −1`, so treat `accuracy < 0` as invalid [1].
  - **True north** needs the magnetic declination for the location (WMM2025 evaluated at the Geolocation fix). Implemented: `src/sensors/wmm.ts` + `CompassSensor.setLocationFromFix` (app) and the world alignment's own conversion of magnetic-flagged headings (ARCHITECTURE §6 "True-north heading").
- **`webkitCompassAccuracy` [V].** `headingAccuracy` is the maximum deviation in degrees from true magnetic heading. Negative means invalid or uncalibrated [21].
- **Alignment.** Combine the precise relative `alpha` with the noisy absolute compass by estimating a slowly varying offset `ψ = (webkitCompassHeading + alpha) mod 360`. For heading of the device top: `heading ≈ ψ − alpha`. Average with circular statistics, weight by 1/accuracy², and reject `accuracy < 0` or `> 25°` [U thresholds].
- **iOS 26 heading change [U].** A native-app report (Mapillary, May 2026) says iOS 26 now compensates heading for **landscape interface orientation** at OS level. On iOS ≤ 18 the app had to add 90° itself, so on iOS 26 the correction was applied twice [36]. It is unknown whether this affects `webkitCompassHeading`, whose `CLLocationManager` lives in Safari's UI process.
  - **Fallback:** never hard-code a landscape correction. Check whether `ψ` stays constant across an orientation change (§7, check E), and if it jumps by about ±90°, adjust per θ.
  - Near β ≈ 90° (phone upright, the normal capture pose), the projection of the device-top direction is degenerate. Use compass only as a soft prior on world yaw, with σ ≥ 10–15°.

---

## 3. Generic Sensor API

**None of it is available on iOS Safari [V].** BCD lists `Accelerometer`, `Gyroscope`, `LinearAccelerationSensor`, `AbsoluteOrientationSensor`, `RelativeOrientationSensor`, `GravitySensor` and `Magnetometer` all as `version_added: false` for Safari and iOS Safari, while Chrome Android has them from 67, 91 or 56 [25]. WebKit's response to the Gyroscope request (bug 273960 → standards-positions #347) was that Apple would likely oppose it as "duplicative of what the Device Orientation and Motion specification already provides" [26].

The W3C Generic Sensor CRD (2026) says the spec is implemented only in Chromium [26]. WebXR (`navigator.xr`) is also absent on iOS Safari; it ships only on visionOS [25]. **DeviceMotion and DeviceOrientation are the only inertial sources.**

---

## 4. Geolocation on iOS Safari

| Aspect | Behaviour | Tag |
|---|---|---|
| Provider | `CLLocationManager` in the UI process. `setEnableHighAccuracy:` selects `kCLLocationAccuracyBest` for `true`, otherwise `kCLLocationAccuracyHundredMeters`. **`BestForNavigation` is never used.** `distanceFilter` is not set (none). | [S] [22] |
| Precise vs approximate | `kCLLocationAccuracyBest` is "available only if `isAuthorizedForPreciseLocation`". With Precise Location **off** for Safari Websites, locations are "usually within 1–20 kilometers" and update "at most a few times per hour". **Detect this with `coords.accuracy > ~1000`** and ask the user to enable Settings › Privacy & Security › Location Services › Safari Websites › Precise Location. | [V] [23], [U] settings path |
| Update rate | One position per CoreLocation update, about **1 Hz** from GNSS while moving. It can be slower when stationary or indoors. Each `CLLocation` in a batch fires its own callback. | [S] [22], [U] rate |
| Accuracy | `coords.accuracy` = `horizontalAccuracy` (a radius of uncertainty in metres), **passed through unchanged**. The spec defines it as a **95 % confidence** value, but Apple does not document the confidence level. Typical values are 3–5 m under open sky, 5–20 m next to houses or trees, and 20–65 m indoors. | [S] [22], [V] [24], [U] values |
| `altitude`, `altitudeAccuracy` | Present only when `verticalAccuracy ≥ 0` (WebKit tests `>= 0.0`), otherwise `null`. **`altitude` = `CLLocation.altitude`, which is orthometric (MSL).** Apple says that "in most cases" this means the EGM2008 geoid, and in rare cases the DMA 10×10 grid, which differs by under 5 m. The spec says height above the **WGS84 ellipsoid**, and the difference (geoid undulation) is about −107 to +85 m worldwide. WebKit does not expose `ellipsoidalAltitude`. Typical vertical accuracy is 3–15 m. | [S] [22], [V] [23][24], [U] values |
| `heading`, `speed` | `CLLocation.course` (true north, **direction of travel**, *not* device heading) and `speed` in m/s. Each is `null` when negative (invalid or stationary). | [S] [22], [V] [23] |
| `timestamp` | `CLLocation.timestamp.timeIntervalSince1970` converted to epoch ms. It is "the time at which this location was determined" (the fix time, not delivery time) on the **wall clock**. Convert with `t_perf = timestamp − performance.timeOrigin`, which is approximate because the wall clock can be adjusted. Cached positions can be old, so use `maximumAge: 0`. | [S] [22], [V] [23] |
| `floorLevel` | Non-standard and gated behind a setting (`GeolocationFloorLevelEnabled`). Do not rely on it. | [S] |
| Threads | `Geolocation` / `GeolocationCoordinates` are `Exposed=Window` only, so read them on the main thread and forward to the worker. | [S] |
| Prompts | Two layers of permission: Safari's per-website prompt (default "Ask") and the iOS system authorization for "Safari Websites" (`requestWhenInUseAuthorization`). No gesture is required. Home Screen web apps follow the same Safari Websites system setting. Whether a per-web-app decision persists across launches is [U]. | [S] [22], [U] |
| Permissions API | `navigator.permissions.query({name:"geolocation"})` is mapped in WebKit [S]. **BCD lists it from Safari 16, and iOS mirrors that** [25]. This corrects an earlier draft that said "only recent 26.x". Wrap it in try/catch anyway, because it can reject with `InvalidStateError` when the document has no `Geolocation` object. | [S], [V] [25] |

**Engineering consequences for SfM.**

- GNSS is useless for relative poses at backyard scale (a few metres between frames). Use it for **geo-registration and a weak scale prior** once the trajectory spans ≳ 30–50 m. Weight each fix by `accuracy`, apply Huber or Cauchy loss, and treat `altitude` as MSL; convert to ellipsoidal height with EGM2008 if ECEF is needed.
- The project type `GeoFix.alt` (in `src/core/types.ts`) should be documented as **MSL** on iOS.

---

## 5. Device axes vs rear-camera image axes per screen orientation

**Interface orientation reporting [S][V].**

- `window.orientation` comes from `deviceOrientationForUIInterfaceOrientation(windowScene.effectiveGeometry.interfaceOrientation)` [27]. The mapping is Portrait → 0, PortraitUpsideDown → 180, `UIInterfaceOrientationLandscapeRight` ("Home button on the right", device turned **counter-clockwise** so its top points **left**) → **90**, and `LandscapeLeft` (top points **right**) → **−90** [27][29].
- `screen.orientation.type` = `toScreenOrientationType(window.orientation)`: 90 → `landscape-primary`, −90 → `landscape-secondary` [27]. This was fixed to counter-clockwise semantics in April 2023 (bug 255388, 262940@main), after Safari 16.4 shipped the opposite (bug 254863) [28].
- `screen.orientation.angle` follows the spec's table relative to the natural orientation [27]:
  - **iPhone** (natural portrait): angle = `(window.orientation + 360) % 360`.
  - **iPad** (WebKit treats natural as landscape): angle is 90 for portrait, 0 for `window.orientation` 90, 180 for −90 and 270 for 180. That is **`(90 − θ) mod 360`**, which runs in the opposite direction from iPhone.
- **So derive θ from `window.orientation`.** It is deprecated but still present on iOS, and BCD shows no removal [25]. Fall back to `screen.orientation.angle` on iPhone only.
- `screen.orientation.lock()` is not supported on iOS [25], so the app must handle rotation.

**Camera frame rotation [S].**

- `AVVideoCaptureSource::computeVideoFrameRotation()` sets each `VideoFrame`'s rotation from `m_sensorOrientation − m_deviceOrientation`. `m_deviceOrientation` comes from the page's orientation notifier, which is the same interface orientation [30].
  - An alternative path, `UseAVCaptureDeviceRotationCoordinatorAPI`, would instead take the rotation from AVFoundation's horizon-level (gravity-based) angle. It is an **internal preference that defaults to `false`** except on Apple TV, on `main` and on `safari-7625-branch` [S].
  - If a future Safari turns it on, frames would follow physical orientation even under orientation lock. Check C detects that case, because it shows up as a ±90° error in `R̂_cd`.
- Frames drawn from `<video>` or `createImageBitmap` are therefore **upright for the current UI orientation θ**: image +u points to the screen's right, image +v toward the screen's bottom.
- When **iOS orientation lock** holds the UI in portrait, frames stay portrait-upright even if the phone is physically in landscape. This is consistent, because θ stays 0 as well.
- Safari 27.0 fixed "transient device rotation resulting in captured video frames having the wrong orientation" (180429147) [31]. On 26.x, **drop frames within about 500 ms of an `orientationchange`** and verify the frame aspect ratio (check A in §7).

**Mapping.** θ = device rotation counter-clockwise as seen by the user, which equals `(window.orientation + 360) % 360`. The camera uses the OpenCV frame: x = image right, y = image down, z = optical axis out of the back. The rear camera looks along −z_dev.

```
R_cam←dev(θ) = [[ cosθ, −sinθ,  0],
                [−sinθ, −cosθ,  0],
                [   0 ,    0 , −1]]          v_cam = R_cam←dev · v_dev      (det = +1)
```

| θ | `window.orientation` | iPhone `screen.orientation` | Physical pose (user looking at the screen) | `R_cam←dev` rows | Gravity "down" in the camera frame when held upright |
|---|---|---|---|---|---|
| 0 | 0 | portrait-primary, 0 | portrait, top edge up | (1,0,0) (0,−1,0) (0,0,−1) | (0,+1,0) |
| 90 | 90 | landscape-primary, 90 | top edge to the **left** (UIKit LandscapeRight) | (0,−1,0) (−1,0,0) (0,0,−1) | (0,+1,0) |
| 270 | −90 | landscape-secondary, 270 | top edge to the **right** (UIKit LandscapeLeft) | (0,1,0) (1,0,0) (0,0,−1) | (0,+1,0) |
| 180 | 180 | portrait-secondary, 180 | upside down. Face ID iPhones normally do not rotate the UI into this [U] | (−1,0,0) (0,1,0) (0,0,−1) | (0,+1,0) |

This equals `diag(1,−1,−1)·Rz(θ)` in `docs/ARCHITECTURE.md` §2. Residual camera-to-IMU misalignment and the lever arm (a few millimetres to centimetres) are not zero, but are small. Refine the rotation online with the hand-eye check (check C).

---

## 6. Practical notes from other projects

- **cardboard-vr-display / WebVR polyfill (Google, immersive-web) [V] [33].** This was the most widely deployed web IMU fusion code.
  - It negates `accelerationIncludingGravity` (`accelerometer.set(-x,-y,-z)`) and uses an **OS-specific world rotation**: +90° about x on iOS, −90° elsewhere. This is effectively a fix for iOS's opposite gravity sign.
  - It maps `rotationRate` alpha, beta, gamma directly to x, y, z.
  - It converts deg→rad except on Chrome < 66, which used rad/s.
  - It drops samples whose `timeStamp` goes backwards or whose step falls outside [1 ms, 1 s] ("On Firefox/iOS the `timeStamp` properties can come in out of order").
  - It fell back to `deviceorientation` on iOS **13.4**, where `devicemotion` regressed.
  - It used a postMessage relay for cross-origin iframes (WebKit bug 152299, now handled by Permissions-Policy).
- **Full-Tilt (Rich Tibbett) [V] [16].** The 2014 W3C thread rejected UA-sniffing sign flips because they are "not future compatible". That is why the recipe below *detects* the sign.
- **A-Frame [V] [38].** It ships `device-orientation-permission-ui`: a modal button that calls `requestPermission()` on iOS 13+. Copy that pattern.
- **AR.js location-based [V] [34].** It uses `deviceorientationabsolute` when present, otherwise `deviceorientation` plus `window.orientation`. On iOS that means a relative `alpha`, so world heading is arbitrary unless `webkitCompassHeading` is blended in. Tibbett recommends `360 − webkitCompassHeading` as an absolute alpha substitute on iOS [18].
- **8th Wall [V] [37].** The engine became open source in 2026, but **SLAM ships only as a closed binary**, so its iOS IMU handling cannot be inspected. Its sign and unit fixes are therefore unknown [U].
- **Developers in general [V] [35].** Safari reports `touchstart` is not a user gesture but `touchend` and `click` are. Use HTTPS (`mkcert` locally), or iOS auto-declines.

---

## 7. Normalisation recipe

Output format is the `ImuSample` type in `src/core/types.ts`:

- `t`: ms on the `performance.now()` timeline;
- `gyro`: rad/s, device frame;
- `accel`: linear (kinematic) m/s², spec sign;
- `accelG`: specific force m/s², spec sign, so face-up gives +z.

The device frame is right-handed (x right, y top, z out of screen; portrait-referenced).

```ts
const DEG = Math.PI / 180, G0 = 9.80665;

// ---- 0. Session state (persist nothing across sessions; re-detect each time) ----
let accelSign: 1 | -1 | 0 = 0;          // 0 = not yet determined
const ACCEL_SIGN_PRIOR = isWebKitIOS() ? -1 : +1;   // prior ONLY, until check B decides
let gyroMap = I3;                        // signed permutation, refined by check D (expected identity)
let lastKey = '', firstEventSeen = false;
let lastOri: { t: number; a: number; b: number; g: number; heading?: number; hAcc?: number } | null = null;

// ---- 1. Permission: call synchronously inside a click/touchend handler, BEFORE any await ----
function onStartTap() {
  const req = (window.DeviceMotionEvent as any)?.requestPermission;
  const p: Promise<string> = typeof req === 'function' ? req.call(DeviceMotionEvent) : Promise.resolve('granted');
  // camera + geolocation do not need the gesture; start them after this line
  p.then(s => s === 'granted' ? attach() : showDenied())
   .catch(e => e.name === 'NotAllowedError' ? showStartButton() : showError(e));
}
function attach() {          // attach ONCE and never remove during a session (removal resets alpha reference)
  addEventListener('devicemotion', onMotion);
  addEventListener('deviceorientation', onOrientation);
}

// ---- 2. Raw → normalised sample ----
function onMotion(e: DeviceMotionEvent) {
  if (!firstEventSeen) { firstEventSeen = true; return; }        // WebKit replays last cached sample on attach
  const r = e.rotationRate, ag = e.accelerationIncludingGravity, al = e.acceleration;  // read each once
  if (!r || r.alpha == null) return;                             // no gyro → unusable for VIO
  const key = `${r.alpha},${r.beta},${r.gamma},${ag?.x},${ag?.y},${ag?.z}`;
  if (key === lastKey) return;                                   // duplicate poll of same CoreMotion sample (safe on WebKit: values are unquantised, §1.4)
  lastKey = key;

  let iv = e.interval ?? NaN; if (iv > 0 && iv < 1) iv *= 1000;  // iOS: seconds → ms (informational only)

  // rotationRate: alpha=ω_x, beta=ω_y, gamma=ω_z (spec + WebKit), deg/s → rad/s
  const gyro = mul3(gyroMap, [r.alpha * DEG, r.beta! * DEG, r.gamma! * DEG]);

  const s = accelSign || ACCEL_SIGN_PRIOR;                       // iOS expected −1
  const accelG = ag && ag.x != null ? [s * ag.x, s * ag.y!, s * ag.z!] : null;   // specific force, +z face-up
  const accel  = al && al.x != null ? [s * al.x, s * al.y!, s * al.z!] : null;   // kinematic accel
  const raw = ag && ag.x != null ? [ag.x, ag.y!, ag.z!] : null;

  push({ t: e.timeStamp, gyro, accel, accelG });                 // e.timeStamp: perf.now() ms (1 ms / 20 µs COI)
  feedChecks(e.timeStamp, gyro, raw, al);
}

function onOrientation(e: DeviceOrientationEvent) {
  if (e.alpha == null) return;
  const anyE = e as any;
  lastOri = { t: e.timeStamp, a: e.alpha, b: e.beta!, g: e.gamma!,
              heading: anyE.webkitCompassAccuracy >= 0 ? anyE.webkitCompassHeading : undefined,
              hAcc: anyE.webkitCompassAccuracy };
  feedOrientationChecks(lastOri);
}

// Device "up" unit vector (opposite to gravity) from spec Euler angles, R = Rz(α)Rx(β)Ry(γ), device→world.
// This is CoreMotion's fused attitude, which does not depend on the accelerometer sign convention.
const upDev = (bDeg: number, gDeg: number) => {
  const b = bDeg * DEG, g = gDeg * DEG;
  return [-Math.cos(b) * Math.sin(g), Math.sin(b), Math.cos(b) * Math.cos(g)];
};
const rotFromEuler = (aDeg, bDeg, gDeg) => mul(Rz(aDeg * DEG), mul(Rx(bDeg * DEG), Ry(gDeg * DEG)));
```

**Run-time self-checks.** Run them continuously during the first seconds, then periodically. Log every verdict, and never hard-fail on one alone.

```ts
// A. Timing sanity (every 2 s window)
//   median(dt) ≈ 16.7 ms (accept 14–20); fraction(dt < 2 ms) < 5 % (bursts → main thread is busy);
//   duplicates dropped < 10 %; max gap < 100 ms. If median ≈ 33 ms → main thread congested: warn.
//   Camera: frame aspect must match θ (θ∈{0,180} ⇒ videoHeight > videoWidth for a 4:3/16:9 sensor);
//   ignore frames within 500 ms after 'orientationchange' (Safari 26 transient-rotation bug).

// B. Accelerometer SIGN (needs ≥ 0.5 s of stillness: |gyro| < 0.05 rad/s and | |raw| − 9.81 | < 0.4)
//   c = mean( dot(raw, upDev(β, γ)) / |raw| )   using the deviceorientation event of the same tick
//   c > +0.9 → accelSign = +1 (spec convention: Chrome/Android)
//   c < −0.9 → accelSign = −1 (expected on iOS 26/27)
//   otherwise inconclusive → keep prior, retry. If B flips the prior, re-sign the samples already buffered.
//   Consistency of `acceleration` (same WebKit code path, should share the sign):
//   mean( dot(raw − acc, upDev) ) ≈ accelSign · 9.81  (|err| < 0.5) — else treat `acceleration` as untrusted.
//   Fallback when deviceorientation is missing: user-pose heuristic (UI upright & phone roughly upright ⇒
//   spec-sign accelG·R_dev←cam·(0,−1,0) > 0), only as a weak vote.

// C. Camera↔device rotation (hand-eye). For keyframe pairs (i,j) with visual rotation > 5°:
//   ΔR_dev = Π_k exp([gyro_k·dt_k]×) over [t_i, t_j]   (= R_wd(i)ᵀ·R_wd(j): device(j) → device(i) coords)
//   ΔR_cam = R_i·R_jᵀ  from SfM (world→camera rotations R_i, R_j; maps camera(j) → camera(i) coords)
//   Model: ΔR_cam = R_cd·ΔR_dev·R_cdᵀ  ⇒  r_vis = R_cd·r_imu  with r = log(·) as axis-angle vectors
//   Kabsch over ≥ 5 pairs (r_imu → r_vis) → R̂_cd;  expect angle(R̂_cd, R_cam←dev(θ)) < 3°.
//   ≈ 90°/180° ⇒ wrong θ or image rotation (orientation lock, transient rotation); ≈ 180° about z ⇒ mirrored/flipped.
//   Also: gravity in camera frame g_cam = R_cam←dev·(−upDev); while the user holds the phone upright for the
//   current UI orientation, g_cam·(0,1,0) > 0.7 (warning only, since orientation lock can legitimately violate it).

// D. Gyro axes / sign / units vs. orientation (any time |ω| > 0.5 rad/s)
//   From consecutive orientation events (same CoreMotion sample as the motion events):
//     ΔR = R_kᵀ·R_{k+1};  w_est = log(ΔR)/(t_{k+1} − t_k)            // rad/s, device frame
//   Accumulate C = Σ gyro_raw ⊗ w_est (3×3).
//   Choose the signed permutation P maximising trace(P·C) → gyroMap = P (expected identity).
//   Least-squares slope ≈ 1.0 (≈ 57.3 or 0.0175 ⇒ units bug); |corr| > 0.95 per axis.

// E. Compass (only when 0 ≤ webkitCompassAccuracy ≤ 25)
//   ψ_k = (heading_k + alpha_k) mod 360 should be constant (circular std < 10°) while the user yaws.
//   After an orientationchange, if ψ jumps by ≈ ±90° or 180°, store a per-θ correction (iOS 26 landscape
//   compensation is UNVERIFIED). Use world yaw from compass only as a soft prior (σ ≥ 10–15°).
```

**Time alignment for VIO.**

- Model: IMU physical time = `t − t_d,imu` and camera physical time = `t_frame − t_d,cam`. Estimate the relative offset online, starting from about +15 ms of IMU lag [U].
- The camera timestamp, if available, is `requestVideoFrameCallback` `metadata.captureTime`. In WebKit this is `MonotonicTime::now()` taken when the AVFoundation sample buffer reaches the capture source, mapped onto the same `performance` timeline [30] [S]. It is still later than the exposure time [U: about 10–40 ms].
- Otherwise use `expectedDisplayTime` minus one frame.
- Interpolate the gyro (SLERP on integrated rotation) to the camera timestamps.

---

## 8. UNVERIFIED items and safe fallbacks

| Item | Best estimate | Fallback |
|---|---|---|
| Real event jitter and latency on iPhone 15–17 | About 60 Hz with σ about 2–5 ms; latency about 10–20 ms | Online `t_d` estimation (±50 ms), timing check A |
| Duplicate and dropped sample rate | About 1–5 % | De-duplicate exact repeats; integrate with real dt |
| Timer pauses during UI scroll tracking | Likely | Non-scrolling capture UI |
| Permission persistence across relaunch (Safari and Home Screen web apps) | Re-prompt per launch. WebKit keeps the grant in memory only, but Safari may seed a stored per-site policy through `WKWebpagePreferences` | Always call `requestPermission()` at start; offer a button on `NotAllowedError` |
| Precision quantisation in non-WebKit engines | Possible, since the spec asks for 0.1 units; WebKit/iOS does not quantise [S] | Use exact-repeat de-duplication only on WebKit |
| iPad CoreMotion axes are portrait-referenced | Yes | Checks C and D |
| `webkitCompassHeading` in landscape on iOS 26+ | May now be interface-compensated | Check E per θ |
| GNSS accuracy values | 3–5 m open sky, 5–20 m near buildings | Weight by `accuracy`, robust loss, GPS only for geo-registration |
| Confidence level of `coords.accuracy` on iOS | About 1σ radius (Apple does not define it) | Treat as σ, not 95 % |
| UI rotation to portrait-upside-down on Face ID iPhones | Not available | Handle θ = 180 anyway |
| A future WebKit fix of the accelerometer sign | Not expected | Check B (no UA sniffing) |

---

## References

1. WebKit `Source/WebCore/platform/ios/WebCoreMotionManager.mm`: `main` (2026-10), `safari-7622-branch`, `safari-7625-branch`, and commit `702afb77afff` (2020). https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/ios/WebCoreMotionManager.mm
2. WebKit `Source/WebCore/platform/ios/WebCoreMotionManager.h` (`kMotionUpdateInterval = 1.0f / 60.0f`).
3. WebKit `Source/WebCore/platform/ios/DeviceMotionClientIOS.mm` (interval = `kMotionUpdateInterval`; simulator returns empty data).
4. WebKit `Source/WebKit/UIProcess/ios/WebDeviceOrientationUpdateProviderProxy.mm` (UI-process → web-process IPC, values unchanged).
5. WebKit `Source/WebCore/dom/DeviceMotionEvent.{idl,cpp}` and `DeviceOrientationEvent.idl` (`requestPermission`, the iOS-only `webkitCompassHeading`/`webkitCompassAccuracy`, `absolute` compiled out on iOS); `Source/WebCore/dom/EventNames.json` (no `deviceorientationabsolute`).
6. WebKit `Source/WebCore/dom/DeviceOrientationAndMotionAccessController.cpp` and `Source/WebKit/UIProcess/WebsiteData/WebDeviceOrientationAndMotionAccessController.cpp`.
7. WebKit `Source/WebCore/page/LocalDOMWindow.cpp` (`isAllowedToUseDeviceMotion` / `isAllowedToUseDeviceOrientation`, Permissions-Policy, secure context).
8. WebKit `Source/WebCore/dom/Event.cpp` (`timeStampForBindings`), `Source/WebCore/page/Performance.cpp` (`timePrecision { 1_ms }`), `Source/WebCore/dom/ScriptExecutionContext.cpp` (`setCrossOriginMode` → `allowHighPrecisionTime`), `Source/WTF/wtf/Seconds.h` (`highTimePrecision` = 20 µs), `Source/WTF/wtf/ReducedResolutionSeconds.h` (floor).
9. WebKit `Source/WebCore/page/DeviceController.cpp` (last-event replay on listener add); `Source/WebCore/dom/DeviceMotionController.cpp` (iOS suspend/resume).
10. WebKit bug 145814, "increase DeviceOrientationEvent events emission frequency" (20 Hz → 60 Hz, r185443, 2015-06-10). https://bugs.webkit.org/show_bug.cgi?id=145814
11. WebKit bug 197750 (no re-prompt when the origin already has permission for the session; r245185, 2019). https://bugs.webkit.org/show_bug.cgi?id=197750
12. WebKit bug 201676 (Safari 13: events require `requestPermission()` from a user gesture; Settings toggle removed). https://bugs.webkit.org/show_bug.cgi?id=201676
13. WebKit bug 269706, "DeviceOrientationEvent.requestPermission() fails when called from worker message" (open). https://bugs.webkit.org/show_bug.cgi?id=269706
14. W3C, *Device Orientation and Motion*, Editor's Draft 12 Feb 2025: device frame, rotationRate getters (alpha = x, beta = y, gamma = z, deg/s), interval in ms, requestPermission and transient activation, examples. https://w3c.github.io/deviceorientation/
15. w3c/deviceorientation issue #96, "Opposite z-axis for example 7" (2021; accelerationIncludingGravity face-up = +9.8). https://github.com/w3c/deviceorientation/issues/96
16. R. Tibbett, public-geolocation, 16 Jul 2014 (iOS face-up z = −9.81; vehicle x negative for both fields): https://lists.w3.org/Archives/Public/public-geolocation/2014Jul/0024.html. Follow-ups 31 Oct–1 Nov 2014: https://lists.w3.org/Archives/Public/public-geolocation/2014Nov/0000.html and https://lists.w3.org/Archives/Public/public-geolocation/2014Nov/0002.html
17. public-geolocation, Aug 2017, "update rotationRate alpha, beta, gamma description to be the same as implementation". https://lists.w3.org/Archives/Public/public-geolocation/2017Aug/0000.html
18. public-geolocation, Mar 2014 (iOS `alpha` arbitrary; use `360 − webkitCompassHeading`): https://lists.w3.org/Archives/Public/public-geolocation/2014Mar/0002.html. Jan 2014 gap analysis (old iOS ranges): https://lists.w3.org/Archives/Public/public-geolocation/2014Jan/0000.html
19. Apple CoreMotion docs: `CMMotionManager.attitudeReferenceFrame` (default `xArbitraryZVertical`), `CMAttitudeReferenceFrame.xArbitraryZVertical`, `CMDeviceMotion.gravity` / `.userAcceleration` / `.rotationRate` (bias removed), "Getting raw gyroscope events" (rad/s), "Getting processed device-motion data", `deviceMotionUpdateInterval`. https://developer.apple.com/documentation/coremotion
20. NSHipster, "CMDeviceMotion" (2014, updated 2018): upright portrait `(0, −1, 0)`, flat on its back `(0, 0, −1)`. https://nshipster.com/cmdevicemotion/
21. Apple CoreLocation docs: `CLHeading.magneticHeading`, `CLHeading.headingAccuracy`, `CLLocationManager.headingOrientation` (default portrait top), `headingFilter` (default 1°). https://developer.apple.com/documentation/corelocation/clheading
22. WebKit `Source/WebCore/platform/cocoa/CoreLocationGeolocationProvider.mm`, `Source/WebCore/Modules/geolocation/cocoa/GeolocationPositionDataCocoa.mm`, `Source/WebCore/Modules/geolocation/Geolocation.cpp`, `GeolocationCoordinates.idl`.
23. Apple CoreLocation docs: `CLLocation.altitude` (MSL, EGM2008), `ellipsoidalAltitude`, `course`, `timestamp`, `horizontalAccuracy`; `kCLLocationAccuracyBest`; `kCLLocationAccuracyReduced` ("1–20 kilometers", "a few times per hour"). https://developer.apple.com/documentation/corelocation/cllocation
24. W3C, *Geolocation*: accuracy and altitudeAccuracy at 95 % confidence, altitude relative to WGS84, timestamp = acquisition time, enableHighAccuracy as a hint. https://w3c.github.io/geolocation/
25. MDN browser-compat-data (main, Oct 2026): `api/Accelerometer`, `Gyroscope`, `LinearAccelerationSensor`, `AbsoluteOrientationSensor`, `GravitySensor`, `Magnetometer` (Safari false); `DeviceMotionEvent`, `DeviceOrientationEvent` (`absolute`: iOS false); `Window.orientation`, `deviceorientationabsolute_event`; `ScreenOrientation` (16.4, `lock` false); `XRSystem` (false). https://github.com/mdn/browser-compat-data
26. WebKit bug 273960 → WebKit standards-positions #347 (Gyroscope/Generic Sensor, "duplicative"): https://bugs.webkit.org/show_bug.cgi?id=273960 and https://github.com/WebKit/standards-positions/issues/347. W3C Generic Sensor CRD 2026-05-14: https://www.w3.org/TR/2026/CRD-generic-sensor-20260514/
27. WebKit `Source/WebKit/UIProcess/API/ios/WKWebViewIOS.mm` (`deviceOrientationForUIInterfaceOrientation`), `Source/WebKit/UIProcess/ios/WebPageProxyIOS.mm` (`toScreenOrientationType`), `Source/WebCore/page/ScreenOrientation.cpp` (`angle()`), `Source/WebCore/page/ScreenOrientationType.h` (iPad natural = landscape).
28. WebKit bug 254863, "iOS 16.4 – Screen Orientation API Inconstancy", and bug 255388, "Screen orientation type should be counter-clockwise" (262940@main, 2023-04-13). https://bugs.webkit.org/show_bug.cgi?id=254863 and https://bugs.webkit.org/show_bug.cgi?id=255388
29. Apple UIKit docs: `UIInterfaceOrientation.landscapeRight` ("Home button on the right") and `.landscapeLeft`. https://developer.apple.com/documentation/uikit/uiinterfaceorientation
30. WebKit `Source/WebCore/platform/mediastream/cocoa/AVVideoCaptureSource.mm` (`computeVideoFrameRotation`, `captureTime = MonotonicTime::now()`), `Source/WebCore/platform/graphics/avfoundation/objc/MediaPlayerPrivateMediaStreamAVFObjC.mm`, `Source/WebCore/html/HTMLVideoElement.cpp` (rVFC times → performance timeline).
31. WebKit blog, "WebKit Features for Safari 27.0" (rotation and captured-frame orientation fix 180429147; rAF sub-ms timestamps under COI 176967366). https://webkit.org/blog/18325/webkit-features-for-safari-27-0/
32. WebKit blog, "WebKit Features in Safari 26.0" (every Home Screen site opens as a web app; fingerprinting-script restrictions). https://webkit.org/blog/17333/webkit-features-in-safari-26-0/
33. immersive-web/cardboard-vr-display, `src/sensor-fusion/fusion-pose-sensor.js` and `src/util.js`. https://github.com/immersive-web/cardboard-vr-display
34. AR.js, `three.js/src/location-based/js/device-orientation-controls.js`. https://github.com/AR-js-org/AR.js
35. YellowAfterlife, "Using HTML5 accelerometer and gyroscope in 2020" (gesture rules, `touchend` vs `touchstart`, HTTPS). https://yal.cc/js-device-motion/
36. Mapillary forum, "iOS heading bug" (May 2026, native app, iOS 26 landscape heading compensation). https://forum.mapillary.com/t/ios-heading-bug/10362
37. 8th Wall open-source announcement (SLAM in the binary only). https://8thwall.org/blog/8th-wall-open-source and https://github.com/8thwall/engine
38. A-Frame docs, `device-orientation-permission-ui` component. https://aframe.io/docs/1.4.0/components/device-orientation-permission-ui.html
39. WebKit `Source/WebKit/UIProcess/API/Cocoa/WKWebpagePreferences.mm` (`_setDeviceOrientationAndMotionAccessPolicy:`), `Source/WebCore/dom/DeviceOrientationAndMotionAccessController.cpp` (`accessState` falls back to `DocumentLoader::deviceOrientationAndMotionAccessState()`), `Source/WebCore/Modules/permissions/Permissions.cpp` (`toPermissionName`), `Source/WTF/Scripts/Preferences/UnifiedWebPreferences.yaml` (`UseAVCaptureDeviceRotationCoordinatorAPI`, default false).

---

## Review notes (skeptical review, 2026-10-16)

I spot-checked the most consequential claims against primary sources. I fetched WebKit source from `main` and from `safari-7622/7624/7625-branch` (`raw.githubusercontent.com`), and also checked MDN BCD `main`, the W3C ED of 12 Feb 2025, Apple's CoreLocation/CoreMotion/UIKit docs, the WebKit blog and WebKit Bugzilla.

**Verified, no change needed**

- **Accelerometer sign.** `WebCoreMotionManager.mm` passes `9.80665·userAcceleration` and `9.80665·(userAcceleration + gravity)` through with no sign change. The UI-process proxy and web-process IPC forward the values unchanged. The relevant lines are identical on `main`, 7622, 7624 and 7625.
- **Sign comparison with the spec.** The spec example gives face-up `accelerationIncludingGravity = {0, 0, 9.8}`. CoreMotion reports gravity as the downward vector. The conclusion that iOS reports the negated specific force (and negated kinematic `acceleration`) holds. I also re-derived check B and the `upDev` formula.
- **Rate and `interval`.** `kMotionUpdateInterval = 1.0f/60.0f` drives both the CoreMotion interval and the polling `NSTimer`, and `interval` is that constant in seconds. Bug 145814 confirms 20→60 Hz in r185443, with the 100 Hz and batching requests not landed.
- **`rotationRate`.** It is `rad2deg(x, y, z)` mapped to alpha, beta, gamma.
- **Permission flow.** `mayPrompt = processingUserGesture`. `NotAllowedError` with "requires a user gesture to prompt" is correct. The UI-process `HashMap` is in-memory and keyed by the main-frame origin. Bug 269706 (worker round-trip loses the gesture) is still NEW.
- **`timeStamp`.** It is the event creation time, floored by `ReducedResolutionSeconds::reduce`. It uses 1 ms by default and 20 µs (`Seconds::highTimePrecision`) once `CrossOriginMode::Isolated`. The replay of the last sample on listener add (`DeviceController`) is confirmed.
- **Orientation and camera.** I confirmed the `window.orientation` mapping (`LandscapeRight` = 90 = counter-clockwise), the `toScreenOrientationType` mapping and the iPad natural-landscape `angle()` table, so `(90 − θ) mod 360` is correct. The camera frame rotation from `m_sensorOrientation − m_deviceOrientation` is confirmed, and `R_cam←dev(θ)` checks out for θ = 0 and 90.
- **BCD.** `requestPermission` is iOS 14.5. `DeviceOrientationEvent.absolute` is false on iOS. `Accelerometer` is false. `ScreenOrientation.lock` is false. `Window.orientation` is present on iOS.
- **WebKit blog.** Safari 26.0 says "every website added to the Home Screen opens as a web app". Safari 27.0 (published 2026-09-17) lists 180429147 (captured-frame orientation after transient rotation) and 176967366 (sub-ms rAF under COI).
- **Apple docs.** `CLLocation.altitude` is MSL. `kCLLocationAccuracyReduced` is 1–20 km. `headingFilter` defaults to 1°. `headingAccuracy` < 0 means invalid.
- **Mapillary thread.** May 2026, native app. It is accurately described.

**Changed**

1. **Permissions API for geolocation.** The draft said "iOS support only in recent 26.x [U]". That was wrong: BCD says Safari 16, with iOS mirroring it. Corrected in §4.
2. **Permission persistence.** This was stated as session-only. It is now split into what WebKit does ([S]: in-memory) and what Safari may do on top ([U]: the embedder can seed a policy per navigation through the `WKWebpagePreferences` SPI). I also clarified which cache is keyed by which origin, and that a reload in the same tab may keep events flowing.
3. **Precision (new).** The spec now requires 0.1-unit precision limiting. WebKit iOS applies none [S]. The exact-repeat de-duplication is now scoped to WebKit.
4. **Smaller corrections.**
   - Altitude is present when `verticalAccuracy ≥ 0`, not `> 0`.
   - EGM2008 applies "in most cases", per Apple.
   - The console warning also fires in the not-yet-requested state.
   - A cross-origin iframe without Permissions-Policy resolves `"denied"` rather than rejecting.
   - `permissions.query({name:"accelerometer"})` rejects with `NotSupportedError`.
   - Added a note on the internal `UseAVCaptureDeviceRotationCoordinatorAPI` camera-rotation path, which defaults to off on iOS.

**Still unverified (no device available).** These items remain [U]:
- real latency and jitter;
- duplicate rate;
- timer behaviour during scroll tracking;
- iPad axes;
- whether iOS 26 landscape heading compensation affects `webkitCompassHeading`;
- the confidence level of `coords.accuracy`;
- the 8th Wall and Generic Sensor CRD details, beyond confirming that the URLs resolve.

Also unverified: the mapping from WebKit branch to Safari version (7622 = 26.0–26.1, 7625 = 27.0) is inferred, though 7623 and 7624 have identical sensor code. The face-up sign has been observed on a device only in the 2014 report. Its current status rests on source reading, so keep run-time check B mandatory.
