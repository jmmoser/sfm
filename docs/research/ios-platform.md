# iOS Safari platform constraints for heavy on-device processing, and whether learned features are feasible (iOS 26 / iOS 27)

Status: researched 2026-10-16. Scope: iOS/iPadOS Safari 26.0–26.6 and 27.0 (shipping), plus WebKit `main` where it matters. This doc is a companion to `ios-webgpu.md` (GPU limits) and `ios-camera.md` (capture). Each topic is covered once, in whichever doc owns it.

**How claims are tagged.** The tags match the sibling docs.

- **[V]** Verified in a primary source: Apple release notes, the WebKit blog, a WebKit bug or a W3C spec. MDN browser-compat-data (BCD) 8.1.6, dated 2026-10-15, is also counted as [V].
- **[S]** Read directly from WebKit source on the release branch named. `safari-7625-branch` is Safari 27.0 and `safari-7622-branch` is Safari 26.0. This is strong evidence, but it is still an inference about the shipping binary.
- **[U]** UNVERIFIED. Each one comes with a best estimate and a safe fallback.

---

## 0. Summary table

| Question | Answer | Tag |
|---|---|---|
| WebContent (tab) memory limit, iPhone | Jetsam limit: **about 1.5 GB soft and 2 GB hard**, or on some devices the global Mach task limit. One iPhone 17 Pro (12 GB RAM) was killed at "ActiveHard 2048 MB". The limit is somewhat lower on low-RAM iPhones. **It did not scale with RAM above 8 GB in the one 12 GB report we have.** The WebKit-side `MemoryFootprintMonitor` that quotes these numbers is on WebKit `main` only (not in the Safari 26/27 branches), so in shipping Safari only the OS jetsam limit applies. | [S][12], [V][10][11] |
| WebContent memory limit, iPad | iPads are "desktop class" and have **no per-process WebContent jetsam limit**. Only the Mach task limit (`kern.max_task_pmem`) applies. It is higher on most iPads, but WebKit's own comment warns that *"the task limit on low-RAM iPads may not be large enough"* to look different from an iPhone limit, so do not assume "much higher" on base iPads **[U]**. | [S][12] |
| Practical budget (JS + Wasm + GPU + decoded images, all workers together) | **≤ 1.2 GB on 8/12 GB iPhones, ≤ 0.8–0.9 GB on 6 GB, ≤ 0.6 GB on 4 GB.** Workers and GPU allocations all count against the one WebContent process. | [U], derived |
| Standalone (Home Screen) app memory | Same WebContent process model is expected. Use the same budget. | [U] |
| Max single `ArrayBuffer` | `MAX_ARRAY_BUFFER_SIZE` is **4 GiB** on 64-bit in Safari 26/27. It is 16 GiB on WebKit `main`, which is memory64 prep. In practice, jetsam kills the tab long before either. | [S][14] |
| Max Wasm memory (wasm32) | 65,536 pages = **4 GiB** architecturally. In practice it is bounded by jetsam. `memory.grow` past the budget **kills the tab instead of failing**. Fast memory reserves 4 GiB virtual per memory, and only 3–8 fast memories are allowed. memory64 is not shipped. | [S][16], [V][15] |
| `navigator.deviceMemory` / `performance.memory` | **Neither exists**, so RAM cannot be queried. On 26.2+, `GPUAdapter.limits.maxBufferSize` is a coarse RAM proxy (see `ios-webgpu.md`). | [V][21] |
| Origin storage quota | **60 % of total disk** per origin and 80 % overall for Safari. **Home Screen web apps get the same.** In-app WKWebViews get 15 % / 20 %. There is no prompt: exceeding it throws `QuotaExceededError`. | [V][1], [S][19] |
| Eviction | Per origin, LRU, triggered when the overall quota is exceeded or the device is under storage pressure. Origins that are persisted or have an active page are excluded. | [V][1] |
| 7-day cap (Safari tab) | After **7 days of Safari use with no user interaction on the site**, all script-writable storage is deleted: IDB, LocalStorage, SW registrations, Cache, and also OPFS according to the source. **Home Screen web apps are exempt.** | [V][2], [S][17][18] |
| `navigator.storage.persist()` | **Granted only if the domain is in WebKit's "exempt from website-data deletion" set.** That set is the app-bound domains, MDM-managed domains, Safari's `_persistedSites` (an SPI since iOS 18.2), and **the standalone web app's own domain**. So it should be **true in a Home Screen app**. In a Safari tab it is usually false. | [S][17][18]; tab behaviour [U] |
| Home Screen app storage vs Safari storage | **Separate containers, by design.** Data captured in a Safari tab does not appear in the installed app. | [V][20] |
| OPFS | `getDirectory` and `createSyncAccessHandle` (dedicated worker only) from **15.2**. Synchronous `getSize/flush/close/truncate` from **16.4**. `createWritable` / `FileSystemWritableFileStream` from **26.0**. `removeEntry()` and `resolve()` were fixed in **27.0**. There is no `readwrite-unsafe` mode. | [V][3][5][9][21], [V][43] |
| IndexedDB reliability | Long history of "Connection to Indexed Database server lost" errors. **26.5** fixed connections becoming permanently broken. **27.0** fixed recovery in workers after a network-process crash, and fixed long blocking when another page's transaction was suspended. | [V][8][9][42] |
| `<a download>` + Blob URL | **Works in a Safari tab.** It shows a download prompt and saves to Files ▸ Downloads. **Broken UX in Home Screen apps**: a preview sheet or a "navigation-less" state, tracked in bugs 275288, 290847 and 236943. 26.5 fixed file extensions. | [V][24][8] |
| `navigator.share({files})` | Yes, from iOS 14. Default-on. **WebKit has no MIME or extension allowlist**: `canShare` is true for any non-empty `files`, so `.ply`, `.glb`, `.zip` and `.usdz` are all allowed. It needs a **5 s transient activation**. **All files are read fully into memory** before the sheet opens. | [V][21], [S][22][23] |
| `showSaveFilePicker` / `showOpenFilePicker` / `showDirectoryPicker` | **Not supported.** | [V][21] |
| Module workers | Safari **15**. Nested workers 15.5, with script loading in nested workers from 16.4. | [V][21] |
| `navigator.hardwareConcurrency` | Returns **4 if the device has fewer than 8 cores, otherwise 8**. **Every iPhone (6-core A15–A19) reports 4.** M-series iPads report 8. On Safari 26.x/27.0, scripts flagged as known fingerprinters get a seeded random value from **1 to 63**; WebKit `main` changes that to a fixed 4 or 8. First-party code is unaffected. | [S][25], [V][21] |
| OffscreenCanvas | 2D from 16.4. WebGL/WebGL2 in OffscreenCanvas from 17. `"webgpu"` from 26. `transferControlToOffscreen` from 16.4. | [V][21] |
| Transferables | Transferable: ArrayBuffer, **ImageBitmap** (also clonable), **OffscreenCanvas**, **VideoFrame** (serialize/transfer), **MediaStreamTrack** (18), **ReadableStream (27)**, FileSystemHandle (serializable). **Not** transferable: Writable/TransformStream. | [S][26], [V][21][9] |
| onnxruntime-web + WebGPU on iOS | **Unreliable through Safari 26.x.** The JSEP and asyncify builds trigger a runaway in WebKit's Wasm OMG compiler (register/stack allocation on huge asyncify-inflated functions) on 26.2+, with CPU at 400 %+ and multi-GB memory, ending in jetsam. WebKit landed allocator improvements (306945@main and related fixed bugs) in Feb–Mar 2026, but WebKit 304810 is still **NEW** and a 2026-09-15 comment reproduces the jetsam on **iOS 26.6.2** (4 of 4 runs). ORT #26827 was closed by a stale bot, not fixed. Whether Safari 27 fixes the asyncify/JSEP case is **[U]**. The **JSPI build** (`onnxruntime-web/jspi`, ORT 1.30) needs **Safari 27**, which shipped JSPI, and is **untested on iOS**. | [V][30][31][32][9], [S][33]; Safari 27 + JSPI path [U] |
| Primary feature pipeline | **Classic GPU SIFT (hand-written WGSL)**. Learned features should be an **optional plug-in**, preferably XFeat hand-ported to WGSL with no ORT dependency. | Recommendation |
| SW + manifest | Service workers and the Cache API from 11.3. `display: standalone` from 11.3. `fullscreen`, `minimal-ui`, `orientation`, `share_target`, `file_handlers` and `beforeinstallprompt` are not supported. Since **26.0, any site added to the Home Screen opens as a web app by default**. | [V][21][5][4] |
| Camera permission in Home Screen app | **Reported to be asked again on every launch.** Bug 327237 (NEW, 2026-10-12, filed by a third party against iOS 26.6.2; imported to Radar, no WebKit triage yet) covers iOS 26/27. In a Safari tab, a grant is reused for **10 min after capture stops** and **24 h during capture** (WebKit defaults), unless the user sets the site's Camera permission to "Allow". **On iPhone, a `getUserMedia` call made without a user gesture reuses the grant only if capture stopped less than 1 min ago.** | [V][27] (report only), [S][28] |
| Motion permission | `DeviceMotionEvent.requestPermission()` needs a user gesture. WebKit **caches the decision in memory only**, so expect to ask again on each launch, and a denial sticks until relaunch. The embedder can pre-seed the cache (`setCachedDeviceOrientationPermission`), so whether Safari restores it from its own per-site settings is **[U]**. Since **26.4** the interfaces exist only in secure contexts. | [S][29], [V][7] |
| HTTPS | Required for camera, motion (26.4+), WebGPU, SW, OPFS/StorageManager, Web Share, Wake Lock and SAB. SAB additionally needs COOP `same-origin` + COEP `require-corp`. | [V][21][7] |
| Lockdown Mode | Disables **IndexedDB, OPFS, Service Workers, Cache API, WebGPU, WebGL, MediaDevices (camera), WebCodecs video, Web Share and FileReader**. The app cannot work. Detect it and tell the user to exclude the site. | [S][23] |

---

## 1. Memory

### 1.1 When WebKit or iOS kills the tab

- **Primary statement [S].** WebKit `main` (`MemoryFootprintMonitor.cpp`, PR #75688, 2026-10-06) says: *"On iOS, the jetsam limit can be 1.5GB (soft), 2GB (hard), or global Mach task limit based on device."* It also says: *"Desktop class devices (e.g. iPads) don't impose a per-process jetsam limit on WebContent."* WebKit's own monitor defaults to a **2 GB** page limit on iPhone, polled every 30 s [12].
  - **Scope (review correction).** `MemoryFootprintMonitor.cpp` exists only on `main`. It is absent from `safari-7622-branch` and `safari-7625-branch` and is not listed in their `SourcesCocoa.txt`. The comment is good evidence about the **OS** limits, but the WebKit-side 2 GB page kill is **not** in Safari 26.x or 27.0. It may ship later, and would kill at about the same level the OS already does.
  - PR #75688 adds a heuristic: WebContent reports its jetsam limit at start-up, and a limit **above 3 GB** is treated as "no per-process jetsam limit". This means some devices run WebContent under the task limit only. Which iPhones, if any, do so is **[U]**.
- **Field data [V].**
  - Simon Fraser (WebKit, 2024): *"WebContent processes on iPhone are subject to a 'soft' limit of around 1.5GB … It will be somewhat lower on iPhone models with less RAM"* [10].
  - Chris Dumez (2021): a jetsam at *"over 1.5GB"* [13].
  - Apple Developer Forums, April 2026: an iPhone 17 Pro with **12 GB** RAM on iOS 26.3–26.4.1 logged `WebKit.WebContent … ActiveHard 2048 MB (fatal)`. An Apple engineer replied: *"iPhones do indeed have stricter jetsam limits than iPads"* [11].
- **Soft vs hard.** Above the soft limit the process *may* be killed when the system is under pressure, for example when the camera, the GPU process or other apps are busy. Above the hard limit the kill is certain. Capture plus processing is exactly the high-pressure case, so **treat the soft limit as the ceiling**.
- **Estimates by device class [U].**

  | iPhone RAM | Example models | Expected WebContent ceiling | Budget to design for |
  |---|---|---|---|
  | 4 GB | iPhone 12/13 (non-Pro) | about 1.0–1.4 GB | ≤ 0.6 GB |
  | 6 GB | 12 Pro–13 Pro, 14, 15 | about 1.4–1.5 GB soft | ≤ 0.8–0.9 GB |
  | 8 GB | 15 Pro, 16 family, 17 | 1.5 GB soft / 2 GB hard | ≤ 1.2 GB |
  | 12 GB | 17 Pro, Air | 1.5 GB soft / **2048 MB hard (observed)** | ≤ 1.2–1.4 GB |

- **What counts toward the limit [S].** One WebContent process holds the main thread, **all dedicated workers**, the JS heap, Wasm memories, canvas backing stores, decoded images, and in-flight VideoFrames and ImageBitmaps. Per `ios-webgpu.md` §4.2, **GPU buffers and textures are also attributed to WebContent**.
- **There is no warning.** A kill shows *"A problem repeatedly occurred"* or a silent reload. JS gets no event.
  - Persist a "processing in progress" marker plus checkpoints to OPFS.
  - On the next load, offer to resume with a lower-memory profile.
- **Detecting RAM.** `navigator.deviceMemory` and `performance.memory` are absent [21]. On 26.2+, `adapter.limits.maxBufferSize` is RAM-dependent: roughly `min(1 GiB, maxBufferLength/3)`, where `maxBufferLength` is about 23–25 % of RAM. Use it to choose 4 / 6 / 8 GB+ profiles (`ios-webgpu.md` §2).

### 1.2 ArrayBuffer and typed arrays

- **Hard cap [S].** `MAX_ARRAY_BUFFER_SIZE = 1ull << 32` (4 GiB) under `USE(LARGE_TYPED_ARRAYS)`, which is on for all 64-bit builds, on both `safari-7622-branch` and `safari-7625-branch`. WebKit `main` raises it to `1ull << 34` (16 GiB) [14].
- **Related APIs [V].** Resizable buffers and `maxByteLength` from 16.4. `ArrayBuffer.prototype.transfer()` from 17.4 [21].
- **Practical limits [U].**
  - Avoid single allocations above about 256–512 MB. One allocation is subject to the same jetsam budget and to address-space fragmentation.
  - Large JS allocations are also the most likely trigger for an instant kill.
  - The WebGPU per-buffer limit (256 MiB on 26.0–26.1, up to 1 GiB on 26.2+) is the tighter bound for GPU-resident data anyway.

### 1.3 WebAssembly memory

- **Maximum size [S][V].** wasm32 can reach 65,536 pages (4 GiB), bounded by `MAX_ARRAY_BUFFER_SIZE` (`WasmMemory.cpp`) [14][16]. memory64 and multi-memory are in STP only (BCD: `false` for Safari iOS) [21].
- **Fast memory [S].** `useWasmFastMemory` reserves **4 GiB of virtual address space plus a redzone** for each memory. `maxNumWasmFastMemories` is 8, or 3 without a large Gigacage. Memories beyond that fall back to explicit bounds checks [16]. **Use one shared, pre-sized Wasm memory, not many instances.**
- **Failure mode [V].** `memory.grow` does not fail cleanly: *"Safari kills/crashes/reopens the whole tab without notice"* [15].
  - In 2024 the Gigacage fragmentation "MegaBug" (269937) reported only about 300–500 MB usable after a reload.
  - It was resolved by enlarging the Gigacage (272171) and enabling Wasm fast memory on iOS (272232). The release that shipped the fix is not stated; probably Safari 18 [U].
- **Rule.** Declare a modest `maximum` (512 MiB–1 GiB on 8 GB phones) and size the memory once at start-up. Never grow it in a loop.

---

## 2. Storage

### 2.1 Quota and eviction (Safari 17+, unchanged through 27.0)

- **Quota [V].** WebKit's *Updates to Storage Policy* post (Safari 17) gives the numbers [1]:
  - **Origin quota:** up to **60 %** of total disk for browser apps and 15 % for other apps.
  - **Overall quota:** **80 %** for browser apps and 20 % for others.
  - Cross-origin frames get 10 % of the main frame's origin quota.
  - *"When a web app is running standalone (Home Screen Web App) … it has the same origin quota and overall quota as when it is opened in a browser app."*
  - Source: `WebsiteDataStoreCocoa.mm` chooses browser or app ratios via `isFullWebBrowserOrRunningTest()`. `NetworkStorageManager` multiplies by **total volume capacity rounded up to whole GB**, not by free space [19][17].
- **Practical consequence.** On a 256 GB phone the nominal quota is about 150 GB, but **free disk is the real limit**. Always check `navigator.storage.estimate()` (17.0+) before a capture session and handle `QuotaExceededError` on every write. Since 17 there is no prompt [1].
- **Eviction [V].** Eviction is per origin and LRU, and happens when the overall quota is exceeded, under system storage pressure, or after an ITP inactivity period. *"Origins with active pages or persistent storage are excluded"* [1].
- **7-day cap [V].** This has applied since Safari 13.1 / iOS 13.4. In a Safari tab, after **seven days of Safari use without user interaction on the site**, WebKit deletes all script-writable storage: IndexedDB, LocalStorage, SessionStorage, media keys, and SW registrations and cache.
  - **Home Screen web apps** *"have their own counter … we do not expect the first-party in such a web application to have its website data deleted"* [2].
  - In source, `domainsExemptFromWebsiteDataDeletion()` includes `m_standaloneApplicationDomain` [18].
  - ITP's "delete all script-written storage" goes through `NetworkStorageManager`, which removes all managed types for the domain, including the File System type. **OPFS should therefore be treated as subject to the 7-day cap in a tab** [S, inferred].
- **New in the 27.0 branch [S].** `NetworkStorageManager::performTimeBasedEviction` adds a 7-day sweep. It is disabled whenever tracking prevention is on, which is Safari's default, so it matters only for full-browser apps with ITP off [17]. Persisted and active origins are skipped.

### 2.2 `navigator.storage.persist()`

- **Mechanism [S].** `NetworkStorageManager::persist()` returns true **only if** the origin's registrable domain is in the exempt set, and only for first-party origins. In that case it writes a `persisted` marker file [17]. The exempt set is [18]:
  - app-bound domains;
  - managed (MDM) domains;
  - "persisted domains", which Safari supplies through the `_persistedSites` SPI (iOS 18.2+);
  - **the standalone application domain**.
- **So:**
  - In a **Home Screen web app**, `persist()` should resolve `true`.
  - In a **Safari tab** it is usually `false`. Which sites Safari puts in `_persistedSites` is closed-source **[U]**.
  - The WebKit blog describes this as *"heuristics, such as whether the site is opened as a Home Screen Web App"* [1].
- **Rule.** Call `persist()` on every launch in standalone mode and record the result. In a tab, treat storage as best-effort and **push the user to install** before a long capture.

### 2.3 Home Screen app storage is a separate container

- **Behaviour [V].** WebKit bug 181849 says Home Screen apps don't share storage with Safari. WebKit's position is *"by design. Home Screen apps are created as isolated entities"* [20].
- **Consequences:**
  - Never let a user capture a large project in a Safari tab and then install. The project will not be visible in the app.
  - Show the install prompt *before* the first capture.
  - Offer an export/import (`.zip`) path between the two.
- Since **26.0**, any site added to the Home Screen opens as a web app by default; users can switch off "Open as Web App" [5][4].

### 2.4 OPFS

- **Version history [V].**
  - `navigator.storage.getDirectory()`, directory and file handles, `getFile()`, `move()`, `removeEntry()` and `createSyncAccessHandle()`: **15.2**. `getFile` landed in 15.4 per the blog [3][21].
  - `FileSystemSyncAccessHandle.getSize/flush/close/truncate` were **promise-returning until 16.4**; they are synchronous from 16.4. `read`/`write` were synchronous from 15.2 [21].
  - `createWritable()` / `FileSystemWritableFileStream`: **26.0** (145875384) [5][21].
  - **27.0 fixed** `removeEntry()` to correctly remove entries (175745157) and `resolve()` paths (175645387) [9].
- **Behaviour [V].**
  - Sync access handles exist **only in dedicated workers**.
  - Only one handle per file can be open. WebKit has no `readwrite-unsafe` mode (bug 283959), so one file can't be shared across workers [3][43].
  - There was a historical iOS regression, `UnknownError: Invalid platform file handle` (bug 251460, fixed) [43].
- **Restrictions [V]/[S].** OPFS is unavailable in Private Browsing [3]. That is from 2022 and is **[U] for 26/27**. It is disabled in Lockdown Mode (`FileSystemEnabled` has `disableInLockdownMode`) [23].
- **Handles can be posted [S].** `FileSystemHandle` is serializable (`FileSystemHandleTag` in `SerializedScriptValue`), so directory handles can be posted to workers [26].
- **Before 27.0 [U].** Deleted entries may not be removed correctly. As a fallback, `truncate(0)` the file through a sync handle and close it before calling `removeEntry()`. Then verify with `estimate()`.

### 2.5 IndexedDB and Blobs

- **Blob support.** Blobs and Files can be stored in IDB. WebKit keeps large values as separate files on disk. The exact threshold is [U].
- **Reliability fixes [V].**
  - **26.4:** fixed mismatched metadata and name-encoding versions (163219457) [7].
  - **26.5:** fixed *"IndexedDB connections could become permanently broken until the page was reloaded"* (172247569) [8].
  - **27.0:** fixed version-0 databases after an aborted first upgrade (176195526); *"IndexedDB connections in workers to recover after a network process crash"* (177219395); and long blocking when *"another page's transaction was suspended in the background"* (178769599) [9].
- **Historical bugs [V].** Bugs 197050, 235579 and 273827 report "Connection to Indexed Database server lost" after backgrounding, up to iOS 17.5 [42].
- **Rule.** Wrap IDB in a small layer that:
  - reopens the database on `UnknownError` or `InvalidStateError`;
  - keeps transactions short;
  - never holds a transaction open across an `await` on non-IDB work.

### 2.6 Storing many multi-MB JPEG frames (recommended layout)

- **Image bytes go to OPFS through a SyncAccessHandle in the storage worker.**
  - Use one file per frame (`frames/000123.jpg`), or append-only pack files of about 64 MB plus an offset index.
  - Sync writes are the fastest path in WebKit [3]. They avoid structured-cloning Blobs through IDB and keep JPEG bytes out of the JS heap.
- **Metadata goes to IndexedDB, or a binary manifest in OPFS.** This covers timestamps, IMU samples, intrinsics, initial poses and the feature-file index. Keep features, matches and depth maps in OPFS binary files as well.
- **Size budget.** A 12 MP JPEG at quality 0.9 is about 2.5–4 MB. 300 frames take about 1 GB, which is fine against the quota.
  - Check `estimate()` and stop capture at roughly 80 % of `quota - usage`, or below about 2 GB of free space. **[U]:** free disk is not exposed directly.
- **Encoding.** `ImageCapture.takePhoto()` already returns a JPEG Blob (see `ios-camera.md`). For frames from `VideoFrame`, use `OffscreenCanvas.convertToBlob({type:'image/jpeg', quality})` in a worker (16.4+) [21].
- **Decoding.** Use `createImageBitmap(blob)` in the worker (Safari 15). `close()` every ImageBitmap and VideoFrame promptly, because their backing stores count against jetsam.

---

## 3. File export and download

| Path | Safari tab | Home Screen app | Notes |
|---|---|---|---|
| `<a href=blobURL download="scan.glb">` | **Works.** A prompt appears, then Files ▸ Downloads. Bug 216918 confirmed blob downloads work on iOS Safari (17.3) [24]. | **Unreliable.** It shows a preview instead of saving (275288, closed as Safari UI [24]), or leaves a "navigation-less state" that needs a force-quit (290847, NEW, iOS 18.4 [24]; 236943 MOVED [24]). | 26.2 fixed "service worker downloads not being saved to Downloads" (154501503) [6]. 26.5 fixed wrong or missing file extensions (173705083, 173945210) [8]. Set a correct `Blob.type` and an explicit extension. |
| `navigator.share({files})` | Works (iOS 14+) [21] | **Works. Preferred in standalone mode.** | Details below. |
| `showSaveFilePicker()` | **No** [21] | No | `createWritable` (26.0) applies to OPFS handles only. |
| AR Quick Look (`<a rel="ar" href="x.usdz">`) | [U] with blob URLs | [U] | Optional iOS-native viewer. Test it before relying on it. |

### Web Share Level 2 details [S][22][23]

- **On by default.** `WebShareFileAPIEnabled` and `WebShareEnabled` default to true on Cocoa (not on watchOS or tvOS). **`WebShareEnabled` is disabled in Lockdown Mode**, which removes `share()` entirely. `WebShareFileAPIEnabled` itself has no Lockdown flag, but that makes no difference in practice.
- **`canShare()` checks no file types.** In `Navigator.cpp`, `canShare()` returns true whenever `files` is non-empty, policy allows it, and any URL is valid.
  - Unlike Chromium, **WebKit has no MIME or extension allowlist**, so `.ply`, `.glb`, `.gltf`, `.obj`, `.zip` and `.usdz` all pass.
  - The share sheet then decides which targets to offer, such as Save to Files, AirDrop, Mail or apps, based on the system type for the extension. Unknown types still get "Save to Files" **[U]**. Give every file a correct name, extension and `type` (`model/gltf-binary`, `application/zip`, `application/octet-stream` for PLY).
- **Activation.** `share()` must consume **transient user activation**, which lasts **5 s** (`defaultTransientActivationDuration { 5_s }`) [22]. Without it the call rejects with `NotAllowedError`.
  - **Build the export file before the tap**, for example behind a "Prepare export…" step. Then call `share()` synchronously in the click handler.
  - A second call while the sheet is open rejects with `InvalidStateError`. Cancelling rejects with `AbortError`.
- **Memory.** Before the sheet opens, `ShareDataReader` reads **every file fully into an ArrayBuffer** and copies it into a `SharedBuffer` for IPC [22].
  - Peak extra memory is therefore about **2× the total file size** inside WebContent, which counts toward jetsam.
  - **Keep each share at or below about 200–300 MB [U].**
  - Compress: GLB with quantization and meshopt or Draco; PLY as binary little-endian; ZIP built in a worker with `CompressionStream('deflate-raw')` (16.4+) [21].
  - Split large point clouds into several files.
- **Large exports from a Safari tab.** Prefer `<a download>` with a URL created from an **OPFS-backed `File`** (`fileHandle.getFile()`). The download path probably streams from disk instead of copying into JS memory **[U]**: test it with a file of 500 MB or more.

---

## 4. Workers, threads, OffscreenCanvas, transfers

- **Module workers [V].** `new Worker(url, {type:'module'})` works from Safari **15**. Nested workers from 15.5, with script loading in nested workers from 16.4 [21]. Safari 27 decodes worker scripts as UTF-8 (175327455) and improves cross-origin-isolation enforcement for workers (176175488) [9].
- **`hardwareConcurrency` [S][25].** `NavigatorBase::hardwareConcurrency` returns `WTF::numberOfProcessorCores() < 8 ? 4 : 8`.
  - **Every current iPhone reports 4.** A15–A19 are all 6-core: 2 performance + 4 efficiency cores (Apple spec).
  - M-series iPads report 8.
  - If the calling script is "tainted" (a known fingerprinting script under Safari 26's protections [4]), the result on `safari-7622-branch` and `safari-7625-branch` is `1 + WeakRandom{seed}.getUint32(63)`, a seeded random number from **1 to 63**. WebKit `main` now returns a fixed 4 or 8 instead. First-party code is unaffected.
  - BCD records the 4-or-8 clamp from Safari 15.4 [21].
- **Pool sizing [U].** 1 orchestrator/WebGPU worker, 2–3 CPU compute workers and 1 storage (OPFS) worker. Only 2 cores are fast, so CPU-heavy steps like bundle adjustment and RANSAC gain little from more than 3 parallel workers, and sustained load causes thermal throttling. Keep heavy math on the GPU.
- **OffscreenCanvas [V].** 2D from 16.4; WebGL/WebGL2 contexts in workers from 17; `"webgpu"` from 26; `transferControlToOffscreen` and `convertToBlob` from 16.4 [21]. `OffscreenCanvasInWorkersEnabled` is on by default on Cocoa [23].
- **Structured clone / transfer [S][26].** `SerializedScriptValue` (27.0 branch) has tags for:
  - `ImageBitmapTransferTag` and `ImageBitmapTag`: **ImageBitmap** can be transferred or cloned;
  - `OffscreenCanvasTransferTag`;
  - `WebCodecsVideoFrameTag`: **VideoFrame** can be posted. The IDL still carries a "FIXME: Support Serializable and Transferable", so post it in the transfer list and `close()` it on the receiving side. The exact semantics are [U];
  - `MediaStreamTrackTag` (18+);
  - `ReadableStreamTag` (**27.0**, 169950592 [9]);
  - `FileSystemHandleTag`;
  - MessagePort.
  - `WritableStream` and `TransformStream` are not transferable [21].
- **SharedArrayBuffer / threads.** These need COOP `same-origin` and COEP `require-corp`; `credentialless` is not supported. Safari 27 fixed SAB cloning and agent-cluster assignment (176465817), and `structuredClone` now throws `DataCloneError` for SAB outside isolation (174562553) [9]. `Atomics.waitAsync` from 16.4 [21]. Details are in `ios-webgpu.md` §6.
- **Backgrounding.** When the page or app goes to the background:
  - workers and timers are suspended (they do not throttle);
  - WebGPU submissions are silently dropped (`ios-webgpu.md` §4.3).
  - Hold a Screen Wake Lock during processing. It works in Safari from 16.4 and in Home Screen apps only from **18.4** [21]. Checkpoint to OPFS at every stage boundary.

---

## 5. Learned local features and matchers on iOS web

### 5.1 State of onnxruntime-web on iOS Safari (October 2026)

- **Builds [S][33].** onnxruntime-web 1.30.0 (2026-09-14) ships three WebGPU-capable bundles:
  - `onnxruntime-web` loads the **JSEP** wasm (28 MB);
  - `onnxruntime-web/webgpu` loads the native WebGPU EP built with **asyncify** (26.8 MB);
  - `onnxruntime-web/jspi` loads the native WebGPU EP built with **JSPI** (16.8 MB).
- **Known failures [V].**
  - **ORT #26827** (Dec 2025 → Sept 2026): in Safari/WebKit **26.2+**, JSEP leaves WebContent at **400 %+ CPU and 1 GB to 14 GB of memory** after inference, and the process is killed on iOS. 26.0 and 26.1 were reported fine [30].
    - Samples show `parseAndCompileOMG()`. The leading hypothesis is WebKit's OMG Wasm JIT struggling with huge asyncify-inflated function bodies. `JSC_useOMGJIT=0` removes the symptom.
    - In Sept 2026 the native WebGPU *asyncify* build was reported to fail the same way on iPadOS 26. The reporter concluded that *"on WebKit without JSPI there is currently **no working GPU path**"* and linked WebKit bug **304810** ("High memory usage in WASM compilation thread", Safari 26.2) [30][32].
    - The issue was closed by a stale bot on 2026-07-18, with **no fix**. Discussion continued into September. Downstream apps pinned ORT 1.20.x, forced the plain `wasm` EP on iOS, or stopped using WebGPU on WebKit. The original report says WebKit 18 also showed the problem "to a lesser extent".
  - **WebKit side (review addition) [V][32].** Bug 304810 is **NEW**, InRadar (rdar 167709015).
    - In Feb 2026 a WebKit engineer wrote that the team was addressing it and the ORT issue *"by improving the greedy register and stack allocators in 306945@main"*. Dependent allocator bugs 307107–309992 are marked FIXED.
    - A duplicate, 309251 (iOS 26 jetsams with IndexedDB writes), was reported fixed in the 26.5 beta.
    - On 2026-09-15 a reporter still saw the WebContent process jetsam-killed on **iOS 26.6.2** (iPhone 16 Pro, 4 of 4 runs), with time spent in `OMGPlan::work` and `allocateRegistersByGreedy`. Peak memory scaled with the number of module instances.
    - The Safari 27.0 release notes list no Wasm compiler-memory fix. The 27.0 branch (`safari-7625`) is newer than 306945@main, so it probably includes those changes, but whether that is enough for ORT is **[U]**.
    - A reporter workaround for self-built asyncify modules is to re-run `wasm-opt -Oz --asyncify`. That does not help with prebuilt ORT bundles.
  - **ORT #27584** (iOS 26.3, ORT 1.24.3, WebGPU EP): a crash after about 500 inferences. There was no maintainer response, and it was closed as stale [31].
- **Safari 27 shipped JSPI** (170260385) [9], so the `/jspi` build becomes plausible. Its behaviour on iOS 27 is **[U]**: no public report was found either way.
- **Operator coverage [V][34].** The WebGPU EP operator table has Conv, Resize, GridSample, MatMul, Softmax, LayerNorm and Einsum. It has **no TopK, no NonZero and no DeformConv**. Keypoint top-k and NMS therefore fall back to CPU with GPU↔CPU copies, and ALIKED's deformable convolutions are not GPU-accelerated.
- **Download and compile cost.** The 17–28 MB wasm plus model weights takes noticeable time on cellular, and Wasm JIT compile on iOS is the code path that is currently fragile.

### 5.2 Candidate models (permissive licences only)

| Model | Licence | Weights (fp32 file) | Output | ONNX availability | iOS-web fit |
|---|---|---|---|---|---|
| **XFeat** (CVPR 2024) | **Apache-2.0** [35] | `xfeat.pt` **6.2 MB** (measured download) | 64-d descriptors. Sparse, plus semi-dense "XFeat*" | Community export PR #5, still open [41]. Trivial to export ourselves. | **Best fit.** Only Conv+BN+ReLU, avg-pool, pixel-unshuffle and bilinear sampling, so it is **easy to hand-port to WGSL with no ORT**. "Real-time on an inexpensive laptop CPU" at VGA. Claimed "up to 5× faster" than prior learned features [35]. |
| **LighterGlue** (XFeat matcher) | Repo Apache-2.0. The weights licence is not stated separately **[U]**. | Not measured. With 6 layers and 96-d (`lighterglue.py`), estimate **≈1–3 MB [U]** | Matches | None official | About 3× faster than LightGlue [35]. MegaDepth-1500 AUC@5/10/20: 0.564/0.710/0.819, versus 0.591/0.738/0.841 for SuperPoint+LightGlue. |
| **ALIKED** (t16 / n16 / n32) | **BSD-3-Clause** [36] | **0.8 / 2.7 / 3.9 MB** (measured) | 64-d (t16) or 128-d [U] | Via LightGlue-ONNX ("RaCo-ALIKED") [38] | Uses `torchvision.ops.deform_conv2d` [37], and DeformConv is not on the WebGPU EP. Hand-porting it is harder. |
| **DISK** | **Apache-2.0** [37][40] | `depth-save.pth` **4.4 MB** (measured) | 128-d | LightGlue-ONNX [38] | A U-Net that is heavy at high resolution. Strong on wide baselines. |
| **LightGlue** (ALIKED / DISK / **SIFT** weights) | **Apache-2.0** for code and weights [37] | **47.6 MB** fp32 per extractor (measured), about 24 MB in fp16 | Matches; adaptive depth and width pruning | LightGlue-ONNX: Apache-2.0, end-to-end export, opset 17, dynamic shapes [38] | Attention is O(n²). Estimate **≈ 90–360 ms per pair at 1k–2k keypoints on A17/A18 [U]**, versus about 10 ms for GPU mutual-NN plus ratio test on 4k SIFT. **`sift_lightglue` can consume our own SIFT** if keypoint scale and orientation conventions match pycolmap [U]. |
| SuperPoint (and SuperPoint-LightGlue) | **Magic Leap "noncommercial research use only"** [39] | — | — | — | **Excluded.** The licence forbids commercial use and distribution. LightGlue's README also notes SuperPoint's restrictive licence [37]. |

### 5.3 Recommendation: classic GPU SIFT first, learned features as a plug-in

**Primary: classic SIFT/DoG implemented directly in WGSL compute.** This means a Gaussian pyramid, DoG extrema, sub-pixel refinement, orientation, 128-d RootSIFT, and GPU brute-force matching with a ratio test and mutual check, followed by geometric verification. The reasons:

1. **Reliability.** It depends only on WebGPU, which has shipped since 26.0. It does not touch the Wasm-JIT path that breaks ORT on 26.2+, and needs no 17–28 MB runtime or model download.
2. **Accuracy for photogrammetry.** It finds full-resolution, sub-pixel, scale-space keypoints on 12 MP frames, which gives low reprojection error in bundle adjustment. Learned extractors usually run at 640–1024 px.
3. **Cost.** Matching cost is predictable, and sequential continuous capture is the easy, small-baseline case.
4. **Determinism and debuggability.** It is easy to validate against COLMAP or pycolmap on the desktop harness.

**Optional plug-in, off by default and enabled per device after a self-test:**

- **XFeat hand-ported to WGSL.** Weights would be fp16, about 3 MB. Use it for low-texture scenes, motion blur and night captures, and as a second opinion for loop closure. Its architecture uses only ops we already write for SIFT and MVS, and it avoids ORT entirely. Estimated 5–15 ms per VGA frame and 20–60 ms at 1024 px on an A17/A18 GPU **[U]**.
- **A learned matcher only for hard pairs.** Use LighterGlue for XFeat, or LightGlue-SIFT for our SIFT. Hard pairs are loop closures, wide baselines, and pairs rejected by the classic matcher. Never use it for all pairs. Run it through either:
  - a WGSL port: about 6–9 transformer layers of MatMul, Softmax and LayerNorm, which is tractable; or
  - **ORT `jspi` on Safari ≥ 27 only**, behind a **crash canary**. Set `sessionStorage['mlInit']=1` before session creation and clear it after N successful runs. If the flag is still set on the next load, disable the ML path permanently for that device and OS version.
- **Never ship** the JSEP or asyncify ORT bundles to WebKit 26.x.

---

## 6. Service worker, PWA manifest, permissions, HTTPS

### 6.1 Service worker and manifest

- **Offline [V].** Service workers and the Cache API have worked since 11.3 [21]. SW and Cache storage count toward the origin quota and are subject to the tab's 7-day cap (§2.1). Safari 27 adds the SW **static routing API** (157951894) and unregisters registrations with missing scripts (174755909, 174833692) [9].
  - Precache the app shell, the WGSL and wasm, and optional model weights. Version the cache and keep weights in their own cache so the shell can update separately.
- **Manifest support [V][21]:**
  - `display: standalone` and `browser`: from 11.3.
  - `fullscreen`, `minimal-ui` and `display_override`: no.
  - `start_url` and `scope`: 11.3. `id`: 16.4.
  - `icons`: 15.4, but only when there is no `apple-touch-icon`.
  - **`orientation`: no. `ScreenOrientation.lock()`: no.** The app cannot lock orientation and must handle rotation during capture.
  - `share_target`, `file_handlers`, `launchQueue` and `beforeinstallprompt`: no.
  - Show custom "Share ▸ Add to Home Screen" instructions. Detect standalone mode with `matchMedia('(display-mode: standalone)')` (12.2+) or the legacy `navigator.standalone`.
- **iOS 26 behaviour [V].** *"Added support for any website to become a web app on iOS or iPadOS"* (113034903) [5]. *"By default, every website added to the Home Screen opens as a web app"* [4].

### 6.2 Camera permission persistence

- **Safari tab [S][28].**
  - `UserMediaPermissionRequestManagerProxy` keeps granted requests per page.
  - A watchdog clears them after **10 min without active capture** (`InactiveMediaCaptureStreamRepromptIntervalInMinutes = 10`) or after **24 h** of continuous capture (`LongRunningMediaCaptureStreamRepromptIntervalInHours = 24`).
  - **Without a user gesture the window is shorter (review addition) [S].** On a call without a gesture, `UserMediaPermissionRequestManagerProxy` reuses a grant only if inactive time is ≤ `InactiveMediaCaptureStreamRepromptWithoutUserGestureIntervalInMinutes`. That value is **1 min** on iOS non-desktop idioms and 10 min on desktop (`WebPreferencesDefaultValues.cpp`). A grant is also not reused while capture is muted. So any re-acquisition of the camera, for example to switch resolution or device, should happen inside a tap.
  - Safari's per-site "Camera: Allow" setting bypasses the prompt. These are WebKit defaults; Safari could override them **[U]**.
- **Home Screen web app [V][27].**
  - A grant is **reported** to last only for the current launch. Bug **327237** ("Home Screen web apps ask for camera/microphone permission again on every launch") was filed 2026-10-12 against iOS 26.6.2, Safari 26/27, and is NEW. It is a third-party report. It was imported to Radar (rdar 189868738), but no WebKit engineer has confirmed the behaviour or its cause, so the iOS 27 behaviour is **[U]**. Design for a prompt on every launch anyway; it is the safe assumption.
  - The older hash-navigation reset (215884) was addressed in 14.5.
- **Mitigation:**
  - Keep the app a single document with no full navigations.
  - Call `getUserMedia` once per launch from an explicit "Start capture" tap.
  - **Keep the track alive** between capture bursts. Toggle `track.enabled` or stop pulling frames instead of calling `stop()`, to stay inside the "active capture" window. Whether a disabled track counts as active is **[U]**.
  - Release the camera only when leaving the capture screen.

### 6.3 Motion and orientation sensors

- **Permission request.** `DeviceMotionEvent.requestPermission()` and `DeviceOrientationEvent.requestPermission()` must be called from a user gesture (BCD: iOS 14.5+) [21]. `Accelerometer`, `Gyroscope` and the other Generic Sensor APIs are **not supported** [21].
- **No persistence in WebKit [S][29].** `WebDeviceOrientationAndMotionAccessController` caches the decision **in memory per WebsiteDataStore**, keyed by top origin. Expect to ask again on every launch. A **denial persists until the app or Safari restarts**, so explain why motion access is needed *before* triggering the prompt.
  - The embedder can pre-seed this cache through `setCachedDeviceOrientationPermission`. Whether Safari restores a decision from its own settings across relaunches is therefore **[U]**; checklist item 9 resolves it.
  - Request camera and motion permission from the same "Start" tap. Call `requestPermission()` first, synchronously inside the gesture, then `getUserMedia`.
- **Secure contexts only [V].** Since **26.4** the `DeviceMotionEvent` and `DeviceOrientationEvent` interfaces exist only in secure contexts (44804273) [7].

### 6.4 HTTPS and headers

- **Secure context required for:** `getUserMedia` (11+), motion and orientation (26.4+), WebGPU, Service Worker, StorageManager/OPFS, Web Share and Wake Lock [21][7].
- **Headers for SharedArrayBuffer and Wasm threads:** `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`. Every cross-origin subresource then needs CORP or CORS.
- **Development on a real iPhone:**
  - `localhost` is not reachable from the phone. Use HTTPS on the LAN with an mkcert root CA, installed as a profile and enabled under Settings ▸ General ▸ About ▸ Certificate Trust Settings.
  - Alternatively, use an HTTPS tunnel that **passes the COOP/COEP headers through**.
  - Debug with Safari Web Inspector over USB. Its Timelines tab shows worker memory, which is improved in 26 [4].

### 6.5 Lockdown Mode [S][23]

- **Disabled features.** `UnifiedWebPreferences.yaml` marks these `disableInLockdownMode: true`: `IndexedDBAPIEnabled`, `FileSystemEnabled`, `FileSystemWritableStreamEnabled`, `ServiceWorkersEnabled`, `CacheAPIEnabled`, `WebGPUEnabled`, `WebGLEnabled`, `MediaDevicesEnabled`, `WebCodecsVideoEnabled`, `WebShareEnabled` and `FileReaderAPIEnabled`.
- **Consequence.** The app cannot function. Detect `!navigator.gpu && !navigator.mediaDevices` and similar, and explain how to exclude the site from Lockdown Mode.

---

## 7. Engineering rules derived from the above

1. **Memory governor.**
   - Keep a global ledger of JS typed arrays, Wasm memory, GPU buffers and textures, ImageBitmaps and VideoFrames.
   - Choose a profile from `maxBufferSize` (RAM proxy) and the UA: 0.6, 0.9 or 1.2 GB.
   - Shrink working sets (tile size, number of frames resident, depth-map resolution) *before* reaching the cap.
   - Destroy GPU resources and `close()` frames explicitly. Never rely on GC.
2. **Crash-safe pipeline.**
   - Every stage writes its outputs to OPFS and updates a manifest atomically: write a temporary file, then `move()`.
   - On load, if a "running" marker exists without a "done" marker, assume jetsam. Resume the stage with a lower-memory profile.
3. **Storage.**
   - Frames and intermediates go in OPFS (sync handles in one storage worker); metadata goes in IDB with a reconnect wrapper.
   - Call `persist()` in standalone mode. Show `estimate()` in the UI.
   - Push installation before the first capture, because storage is separate and subject to the 7-day cap otherwise.
4. **Export.**
   - Prepare the file first (GLB/PLY/ZIP built in a worker with `CompressionStream('deflate-raw')`).
   - On tap: `navigator.canShare({files})` → `share()`. Fall back to `<a download>` in a Safari tab only.
   - Keep each shared file at or below about 200–300 MB.
5. **Threads.** Assume 4 logical CPUs and 2 fast cores. Use 3–5 workers in total, with the GPU doing the heavy lifting.
6. **Learned features.** Classic WGSL SIFT is primary. XFeat-WGSL and a learned matcher are optional and gated. ORT only via `/jspi` on Safari ≥ 27, with a crash canary.
7. **Permissions.**
   - Use one "Start" tap that requests motion permission and then the camera.
   - Run as a single-document app and keep the camera track alive while on the capture screen.
   - Make every later `getUserMedia` call (re-acquire, change resolution or device) from a tap. Without a gesture, iPhone reuses a grant only within 1 min of capture stopping.
   - Expect a camera prompt on each Home Screen launch.

---

## 8. On-device validation checklist (resolves the [U] items)

1. **Jetsam threshold per device class.** In a worker, allocate 64 MB WebGPU buffers and touch them, while logging cumulative size to OPFS. The last value written is the kill point. Test 6, 8 and 12 GB iPhones, in a Safari tab and in a Home Screen app, with the camera on and off.
2. **Wasm memory.** Measure the largest `new WebAssembly.Memory({initial: N, maximum: N, shared: true})` that works after 3 reloads.
3. **Storage persistence.** Record `persist()`, `persisted()` and `estimate()` in a tab and in a Home Screen app, on 26.x and 27.0.
4. **OPFS deletion.** On 26.x, `removeEntry()` a 100 MB file and check that `estimate().usage` drops. Then compare with 27.0.
5. **Share sheet.** Share `.ply`, `.glb`, `.zip` and `.usdz` files of 10, 200 and 500 MB. Record which targets appear (Save to Files?) and the peak memory or crash.
6. **Download.** `<a download>` of an OPFS-backed `File` of 500 MB in a Safari tab: does it succeed without a memory spike? Also test in a Home Screen app on 26.6 and 27.0.
7. **ORT `/jspi` on iOS 27.** Run XFeat and LightGlue (DISK) ONNX for 1,000 inferences while watching memory and CPU. Compare the time per pair with the WGSL implementations.
8. **Camera re-prompt.** In a Home Screen app, record the prompt count across launches, and whether `track.enabled=false` avoids the 10-min re-prompt in a tab.
9. **Motion permission.** Check whether the grant survives relaunch, both in a tab and standalone.
10. **IndexedDB.** Background the app for 5 min mid-transaction on 26.x and 27.0, and verify that the reconnect wrapper recovers.

---

## Sources

1. WebKit blog, *Updates to Storage Policy* (Safari 17; quotas, eviction, `persist()` heuristics): https://webkit.org/blog/14403/updates-to-storage-policy/
2. WebKit blog, *Full Third-Party Cookie Blocking and More* (7-day cap on script-writable storage; Home Screen web app exemption; Safari 13.1 / iOS 13.4): https://webkit.org/blog/10218/full-third-party-cookie-blocking-and-more/
3. WebKit blog, *The File System API with Origin Private File System* (15.2; sync access handle in workers; one handle per file; not in Private Browsing): https://webkit.org/blog/12257/the-file-system-access-api-with-origin-private-file-system/
4. WebKit blog, *News from WWDC25: WebKit in Safari 26 beta* (every site can be a Home Screen web app; File System WritableStream; fingerprinting protections for known scripts, including hardware concurrency; WebGPU): https://webkit.org/blog/16993/news-from-wwdc25-web-technology-coming-this-fall-in-safari-26-beta/
5. Apple, Safari 26.0 release notes (145875384 File System WritableStream; 113034903 any website as a web app): https://developer.apple.com/documentation/safari-release-notes/safari-26-release-notes
6. Apple, Safari 26.2 release notes (154501503 SW downloads to Downloads; Wasm `toResizableBuffer`): https://developer.apple.com/documentation/safari-release-notes/safari-26_2-release-notes
7. Apple, Safari 26.4 release notes (44804273 DeviceMotion/Orientation interfaces only in secure contexts; 163219457 IDB metadata): https://developer.apple.com/documentation/safari-release-notes/safari-26_4-release-notes
8. Apple, Safari 26.5 release notes (172247569 IDB broken connections; 173705083 and 173945210 download extensions): https://developer.apple.com/documentation/safari-release-notes/safari-26_5-release-notes
9. Apple, Safari 27.0 release notes, 2026-09-14 (170260385 JSPI; 175745157 `removeEntry`; 175645387 `resolve`; 177219395, 178769599 and 176195526 IDB; 169950592 ReadableStream transfer; 176465817 and 174562553 SAB; 176175488 worker COI; 157951894 SW static routing): https://developer.apple.com/documentation/safari-release-notes/safari-27-release-notes
10. WebKit bug 277848, comment by Simon Fraser, 2024-08-09 ("soft limit of around 1.5GB … somewhat lower on iPhone models with less RAM"): https://bugs.webkit.org/show_bug.cgi?id=277848
11. Apple Developer Forums 823061, April 2026 (iPhone 17 Pro, 12 GB, "ActiveHard 2048 MB (fatal)"; Apple: "iPhones do indeed have stricter jetsam limits than iPads"): https://developer.apple.com/forums/thread/823061
12. WebKit `main`, `Source/WebKit/UIProcess/MemoryFootprintMonitor.cpp` ("1.5GB (soft), 2GB (hard), or global Mach task limit"; no per-process limit on iPad), and PR #75688 / 322890@main (2026-10-06): https://github.com/WebKit/WebKit/blob/main/Source/WebKit/UIProcess/MemoryFootprintMonitor.cpp, https://github.com/WebKit/WebKit/pull/75688
13. WebKit bug 227636, comment by Chris Dumez, 2021 (jetsam at over 1.5 GB): https://bugs.webkit.org/show_bug.cgi?id=227636
14. WebKit `Source/JavaScriptCore/runtime/PageCount.h` (`MAX_ARRAY_BUFFER_SIZE` = `1ull<<32` on safari-7622 and safari-7625; `1ull<<34` on main), `Source/WTF/wtf/PlatformUse.h` (`USE_LARGE_TYPED_ARRAYS` = CPU(ADDRESS64)), and `wasm/WasmMemory.cpp`: https://github.com/WebKit/WebKit/blob/safari-7625-branch/Source/JavaScriptCore/runtime/PageCount.h
15. WebKit bugs 221530 (`memory.grow` kills the tab), 269777 (about 300 MB reliable after reload), and 269937 (MegaBug, resolved via 272171 Gigacage and 272232 fast memory on iOS): https://bugs.webkit.org/show_bug.cgi?id=221530, https://bugs.webkit.org/show_bug.cgi?id=269777, https://bugs.webkit.org/show_bug.cgi?id=269937
16. WebKit `Source/JavaScriptCore/runtime/OptionsList.h` (`useWasmFastMemory`; 4 GiB virtual reservations plus 128-page redzone; `maxNumWasmFastMemories` 8 or 3): https://github.com/WebKit/WebKit/blob/safari-7625-branch/Source/JavaScriptCore/runtime/OptionsList.h
17. WebKit `Source/WebKit/NetworkProcess/storage/NetworkStorageManager.cpp` on safari-7625-branch (`persist()`, origin quota from volume capacity, time-based eviction): https://github.com/WebKit/WebKit/blob/safari-7625-branch/Source/WebKit/NetworkProcess/storage/NetworkStorageManager.cpp
18. WebKit `Source/WebKit/NetworkProcess/Classifier/ResourceLoadStatisticsStore.cpp` (`domainsExemptFromWebsiteDataDeletion` = app-bound ∪ managed ∪ persisted ∪ standalone app domain), and `WKWebsiteDataStorePrivate.h` (`_persistedSites`, iOS 18.2): https://github.com/WebKit/WebKit/blob/safari-7625-branch/Source/WebKit/NetworkProcess/Classifier/ResourceLoadStatisticsStore.cpp
19. WebKit `Source/WebKit/UIProcess/WebsiteData/Cocoa/WebsiteDataStoreCocoa.mm` (browser vs app quota ratios): https://github.com/WebKit/WebKit/blob/safari-7625-branch/Source/WebKit/UIProcess/WebsiteData/Cocoa/WebsiteDataStoreCocoa.mm
20. WebKit bug 181849, *"Add to homescreen" apps don't share storage with Safari* ("by design … isolated entities"): https://bugs.webkit.org/show_bug.cgi?id=181849
21. MDN browser-compat-data 8.1.6 (2026-10-15), `safari_ios` entries cited inline: https://github.com/mdn/browser-compat-data
22. WebKit `Source/WebCore/page/Navigator.cpp` (`share`/`canShare`), `ShareDataReader.cpp` (files read fully into memory), and `LocalDOMWindow.cpp` (`defaultTransientActivationDuration { 5_s }`) on safari-7625-branch: https://github.com/WebKit/WebKit/blob/safari-7625-branch/Source/WebCore/page/Navigator.cpp
23. WebKit `Source/WTF/Scripts/Preferences/UnifiedWebPreferences.yaml` on safari-7625-branch (default-on flags; `disableInLockdownMode`): https://github.com/WebKit/WebKit/blob/safari-7625-branch/Source/WTF/Scripts/Preferences/UnifiedWebPreferences.yaml
24. WebKit bugs 216918 (blob download works on iOS Safari), 275288 (Home Screen app shows a preview; moved to Safari UI), 290847 (iOS 18.4 navigation-less state; NEW) and 236943 (download link in a PWA cannot be exited; MOVED): https://bugs.webkit.org/show_bug.cgi?id=216918, https://bugs.webkit.org/show_bug.cgi?id=275288, https://bugs.webkit.org/show_bug.cgi?id=290847, https://bugs.webkit.org/show_bug.cgi?id=236943
25. WebKit `Source/WebCore/page/NavigatorBase.cpp` (`hardwareConcurrency`: 4 or 8; random value for tainted scripts): https://github.com/WebKit/WebKit/blob/safari-7625-branch/Source/WebCore/page/NavigatorBase.cpp
26. WebKit `Source/WebCore/bindings/js/SerializedScriptValue.cpp` (transfer and serialize tags), and `WebCodecsVideoFrame.idl`: https://github.com/WebKit/WebKit/blob/safari-7625-branch/Source/WebCore/bindings/js/SerializedScriptValue.cpp
27. WebKit bugs 215884 (hash-navigation reprompt in standalone mode) and 327237 (Home Screen apps ask again on every launch; NEW, 2026-10-12): https://bugs.webkit.org/show_bug.cgi?id=215884, https://bugs.webkit.org/show_bug.cgi?id=327237
28. WebKit `Source/WebKit/UIProcess/UserMediaPermissionRequestManagerProxy.cpp`, plus prefs `InactiveMediaCaptureStreamRepromptIntervalInMinutes` = 10, `LongRunningMediaCaptureStreamRepromptIntervalInHours` = 24, and `InactiveMediaCaptureStreamRepromptWithoutUserGestureIntervalInMinutes` (1 on iOS non-desktop idioms, 10 otherwise, in `Source/WebKit/Shared/WebPreferencesDefaultValues.cpp`): https://github.com/WebKit/WebKit/blob/safari-7625-branch/Source/WebKit/UIProcess/UserMediaPermissionRequestManagerProxy.cpp
29. WebKit `Source/WebKit/UIProcess/WebsiteData/WebDeviceOrientationAndMotionAccessController.cpp` (in-memory decision cache): https://github.com/WebKit/WebKit/blob/safari-7625-branch/Source/WebKit/UIProcess/WebsiteData/WebDeviceOrientationAndMotionAccessController.cpp
30. microsoft/onnxruntime #26827, *Severe CPU and memory issues in Safari/WebKit 26 with JSEP mode* (Dec 2025 – Sept 2026): https://github.com/microsoft/onnxruntime/issues/26827
31. microsoft/onnxruntime #27584, *Crash after around 500 webgpu inference … iOS 26.3 safari*: https://github.com/microsoft/onnxruntime/issues/27584
32. WebKit bug 304810, *High memory usage in WASM compilation thread* (Safari 26.2, asyncify; NEW; allocator work in 306945@main; duplicate 309251; still reproduces on iOS 26.6.2 per comment 11, 2026-09-15): https://bugs.webkit.org/show_bug.cgi?id=304810
33. onnxruntime-web 1.30.0 on npm (2026-09-14). The `dist/` listing has the jsep (28.3 MB), asyncify (26.8 MB), jspi (16.8 MB) and plain (14.2 MB) wasm builds; the `./jspi` export; `ort.webgpu` loads asyncify: https://www.npmjs.com/package/onnxruntime-web, https://unpkg.com/onnxruntime-web@1.30.0/dist/
34. onnxruntime `js/web/docs/webgpu-operators.md` (WebGPU EP operator table): https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md
35. XFeat, verlab/accelerated_features (Apache-2.0; LighterGlue; `modules/lighterglue.py`), and arXiv 2404.19174 (CVPR 2024): https://github.com/verlab/accelerated_features, https://arxiv.org/abs/2404.19174
36. ALIKED, Shiaoming/ALIKED (BSD-3-Clause; t16/n16/n16rot/n32): https://github.com/Shiaoming/ALIKED
37. LightGlue, cvg/LightGlue (Apache-2.0 code and weights; DISK Apache-2.0; ALIKED BSD-3; SuperPoint restrictive; `lightglue/aliked.py` uses `deform_conv2d`; release weights 47.6 MB): https://github.com/cvg/LightGlue
38. LightGlue-ONNX, fabio-sim (Apache-2.0; SuperPoint, DISK and RaCo-ALIKED; end-to-end export; dynamic shapes): https://github.com/fabio-sim/LightGlue-ONNX
39. Magic Leap SuperPoint licence ("ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY"): https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE
40. DISK, cvlab-epfl/disk (Apache-2.0): https://github.com/cvlab-epfl/disk
41. XFeat ONNX export pull request (open): https://github.com/verlab/accelerated_features/pull/5
42. WebKit bugs 197050, 235579 and 273827 ("Connection to Indexed Database server lost"): https://bugs.webkit.org/show_bug.cgi?id=197050, https://bugs.webkit.org/show_bug.cgi?id=273827
43. WebKit bugs 251460 (`createSyncAccessHandle` "Invalid platform file handle" on iOS; fixed) and 283959 (no concurrent OPFS access / `readwrite-unsafe`): https://bugs.webkit.org/show_bug.cgi?id=251460, https://bugs.webkit.org/show_bug.cgi?id=283959
44. Companion docs in this repo: `docs/research/ios-webgpu.md` (GPU limits, GPU memory attribution, backgrounding, COOP/COEP) and `docs/research/ios-camera.md` (capture APIs, `takePhoto`, transfers).

---

## Review notes (skeptical review, 2026-10-16)

A second reviewer spot-checked the claims with the most effect on the design against primary sources, and fixed the errors in place.

**Verified, no change needed.**

- **Safari 27.0 release notes (Apple JSON).** Every bug ID cited from them is present: JSPI 170260385, `removeEntry` 175745157, `resolve` 175645387, ReadableStream transfer 169950592, SAB 176465817 and 174562553, worker COI 176175488, SW static routing 157951894, and IDB 176195526, 177219395 and 178769599.
- **Safari 26.x release notes.** The bug IDs cited from 26.0, 26.2, 26.4 and 26.5 are present as well: 145875384, 113034903, 154501503, 44804273, 163219457, 172247569, 173705083 and 173945210.
- **JavaScriptCore on `safari-7625-branch`.** `useJSPI` is true and `useWasmMemory64` is false. `MAX_ARRAY_BUFFER_SIZE` is `1ull<<32`; it is `1ull<<34` on `main`. `maxNumWasmFastMemories` is 8 or 3.
- **Jetsam evidence.** The Simon Fraser quote (bug 277848, 2024-08-09) and the Apple Developer Forums thread 823061 (iPhone 17 Pro, `ActiveHard 2048 MB (fatal)`, plus the Apple reply) match the sources.
- **Storage quotas.** The 60 %/80 % (browser) and 15 %/20 % (app) quotas and the "same quota when standalone" rule match the WebKit storage-policy post. `NetworkStorageManager::persist()` and `domainsExemptFromWebsiteDataDeletion()` on `safari-7625-branch` match the description, including the standalone app domain.
- **Web Share.** `Navigator::canShare()` has no MIME or extension filter, and `share()` consumes transient activation.
- **Lockdown Mode.** The `disableInLockdownMode` list in `UnifiedWebPreferences.yaml` contains every flag cited.
- **MDN BCD 8.1.6 (published 2026-10-15).** These entries match: `deviceMemory` false; `ScreenOrientation.lock` false; `createWritable` 26; `createSyncAccessHandle` 15.2; OffscreenCanvas `webgpu` 26 and WebGL 17; Wake Lock 16.4, but Home Screen apps only from 18.4; COEP `credentialless` false; motion `requestPermission` 14.5.
- **onnxruntime-web on npm.** 1.30.0 was published 2026-09-14 and has a `./jspi` export.

**Corrected.**

- **`MemoryFootprintMonitor`.** The WebKit-side 2 GB monitor exists only on WebKit `main`. It is not in the `safari-7622` or `safari-7625` branches. The iPad "much higher" limit is qualified, because WebKit's comment warns about low-RAM iPads. PR #75688's "limit above 3 GB" heuristic is added.
- **`hardwareConcurrency` for tainted scripts.** It is 1–63, not 1–64, on 26.x and 27.0. `main` now returns a fixed 4 or 8.
- **Web Share in Lockdown Mode.** Only `WebShareEnabled` carries `disableInLockdownMode`, not both flags. The practical effect is unchanged.
- **ORT and WebKit 304810.** "No fix" was refined. WebKit landed allocator work (306945@main and related bugs), and duplicate 309251 was reported fixed in the 26.5 beta. 304810 is still NEW and reproduces jetsam on iOS 26.6.2 as of 2026-09-15. Safari 27 status is UNVERIFIED.
- **Camera re-prompt.** Bug 327237 is now described as an unconfirmed third-party report. A source-verified detail is added: on iPhone, a `getUserMedia` call without a user gesture reuses a grant only within **1 min** of capture going inactive, versus 10 min with a gesture.
- **Motion permission.** WebKit's in-memory cache can be seeded by the embedder, so whether Safari persists the decision is now marked UNVERIFIED.

**Not re-checked.** The remaining entries were not re-checked and keep the original author's tags:

- the IDB historical bugs;
- the download bugs 275288, 290847 and 236943;
- the XFeat, ALIKED, DISK and LightGlue licences and weight sizes;
- the ORT operator table;
- the Wasm Gigacage history.

The release-branch mapping (`safari-7622` = 26.0, `safari-7625` = 27.0) agrees with the branch list, where 7625 is the newest `safari-762x` branch, but it is not stated anywhere official.
