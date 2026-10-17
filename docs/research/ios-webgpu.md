# WebGPU on iOS Safari (iOS 26 / iOS 27): what an on-device photogrammetry engine can rely on

Status: researched 2026-10-16. Scope: iOS/iPadOS Safari 26.0 through 27.0 (shipping), plus 27.2 beta and Safari Technology Preview (STP) 249–254 where relevant.

**How claims are tagged**

- **[V]** Verified in a primary source: Apple release notes, the WebKit blog, a WebKit bug, a W3C spec, or Apple's Metal tables.
- **[S]** Read directly from WebKit source on the release branch named. This is strong evidence, but it is still an inference about what the shipping binary does.
- **[U]** UNVERIFIED. Each one comes with a best estimate and a safe fallback.

WebKit release branches were matched to Safari versions using `Configurations/Version.xcconfig`. The match is **verified [V]** against the build numbers in Apple's release notes [1][5]: 26.0 = 20622.1.22, 26.1 = 20622.2.11, 26.2 = 20623.1.14, 26.3 = 20623.2.7, 26.4 = 20624.1.16, 27.0 = 20625.1.29, and 27.2 beta = 20625.2.8. Note that the public `safari-7625-branch` tip reads 625.1.27, slightly behind the shipped 27.0 build, so last-minute cherry-picks may be missing from it.

| WebKit branch | WebKit version | Safari |
|---|---|---|
| `safari-7622-branch` | 622.x | 26.0–26.1 (the 26.0.1 build string is 20622.1.22.118.4 [31]) |
| `safari-7623-branch` | 623.x | 26.2–26.3 |
| `safari-7624-branch` | 624.x | 26.4–26.6 |
| `safari-7625-branch` | 625.x | 27.0 (and the 27.2 beta, 625.2.x) |
| `main` | 627.x (commit `e682dc609e75`, 2026-10-16) | STP / future releases |

---

## 0. Summary table

| Question | Answer | Confidence |
|---|---|---|
| WebGPU enabled by default | **Safari 26.0** (iOS/iPadOS 26, Sept 2025). It is not available on older OSes, even Safari 26 on macOS 15. It is disabled in **Lockdown Mode**. | [V] [1][7][31], [S] [21] |
| `navigator.gpu` in DedicatedWorker | **Yes, from 26.0.** IDL is `Exposed=(Window,Worker)`, `WorkerNavigator::gpu()` is implemented, and OffscreenCanvas `"webgpu"` works from 26. | [S] [20], BCD [35] |
| Home Screen web app (standalone) | Expected to work. No gating was found, because `WebGPUEnabled` is an ordinary engine preference. | [U]: feature-detect |
| `maxStorageBufferBindingSize` / `maxBufferSize` | Equal to each other. **26.0–26.1: `max(256 MiB, maxBufferLength/10)`**, so 256 MiB on most phones and possibly about 300 MiB on 12 GB devices. **26.2+: `max(256 MiB, min(1 GiB, MTLDevice.maxBufferLength/3))`**, which is RAM-dependent: roughly 0.45–1 GiB on 6–12 GB iPhones. | formula [S] [16]; values [U] |
| Compute limits (A15–A19) | Workgroup storage 32 KiB. 1024 invocations per workgroup. Size X/Y/Z 1024/1024/1024. 65535 workgroups per dimension. | [S] [16], Metal tables [36] |
| Binding limits | `maxBindGroups` 11. Storage buffers, storage textures and sampled textures are 44 per stage each. Samplers 22. Uniform buffers 44. | [S] [16] |
| Texture limits | 2D 16384. Array layers 2048. 3D 2048. | [S] [16] |
| Above-default limits | **Must be requested.** A device gets *spec defaults* unless `requiredLimits` asks for more (see §2). | [S] [28] |
| `shader-f16` | Yes, 26.0+, all devices. | [S][35][38] |
| `float32-filterable` | **A17 Pro, A18 and A19 only (Apple9/10).** Not on A15/A16 iPhones. | [S] [16] + Metal tables [36]; device check [U] |
| `subgroups` | **No** in 26.x, 27.0 or the 27.2 beta. In STP 249+ (July 2026) and WebKit main. | [V] [10][6], [S] |
| `timestamp-query` | Exposed if Metal stage-boundary timestamp counters exist, which is expected on iPhone. It is costly to use. | [S] [16], [U] on device |
| `bgra8unorm-storage` | Yes, 26.0+. Write-only. | [S] |
| `texture-formats-tier1` / `tier2` | tier1 from 26.2 (BCD agrees). **tier2 from 27.0**, but only in source: BCD says "No" and the 27.0 release notes don't mention it. Feature-detect it. | tier1 [S] [16] + BCD [35]; tier2 [S] only |
| `read_write` storage textures | `r32float`, `r32uint`, `r32sint` from **26.0**. `rgba16float`, `rgba32float`, `rgba8unorm`, `r8unorm` and similar need **tier2, so 27.0+**. | [S] [19] |
| WGSL extensions: `packed_4x8_integer_dot_product`, `pointer_composite_access`, `readonly_and_readwrite_storage_textures`, `unrestricted_pointer_parameters` | **All four from 26.0.** BCD wrongly lists "No". | [S] [18] |
| f32 atomics | **No.** WGSL atomics are `i32`/`u32` only. | spec [39] |
| GPU watchdog duration | Not documented. Long command buffers lead to device loss, or to silently dropped work. | [U]: see §4.1 and §5 |
| Backgrounding | GPU work submitted in the background fails with `NotPermitted`. **The device is not lost**: the work is silently discarded and `onSubmittedWorkDone` still resolves. | [S] [17] |
| GPU out-of-memory during execution | **27.0 only: IOGPU OutOfMemory (code 8) deliberately crashes the GPU process**, so every device in the tab is lost. In 26.x and on main, the same code becomes an ordinary `device.lost`. | [S] [17] |
| `uncapturederror` | **26.x: an `addEventListener` registration delivers at most ONE event** and must be re-armed. There is no `onuncapturederror` attribute until 27.0, which re-arms automatically. | [S] [40], [V] [5] |
| `copyExternalImageToTexture` | Accepts ImageBitmap, ImageData, `<img>`, `<video>`, VideoFrame, canvas and OffscreenCanvas. **Goes through the CPU in 26.x/27.0.** | [S] [22] |
| `importExternalTexture` | Accepts `<video>` and **VideoFrame**. Zero-copy CVPixelBuffer wrap. | [S] [22] |
| WebGL2 as compute fallback | Workable only for fragment-shader GPGPU (no compute, no atomics, no scatter). Use it as a last resort. | [V]/[S] |
| Wasm SIMD + threads | SIMD128 from 16.4. Threads + SharedArrayBuffer from 15.2, and need COOP `same-origin` + COEP **`require-corp`** (no `credentialless`). Relaxed SIMD and memory64 are STP only. JSPI from 27.0. | BCD [35], [V] [9][11][13] |

---

## 1. Availability

- **Shipping version [V].** The Safari 26.0 release notes say "Added support for WebGPU. (145801580)" [1]. The WebKit WWDC25 post says WebGPU ships in Safari 26 on macOS, iOS, iPadOS and visionOS, and "supersedes WebGL" [7].
- **OS requirement [V].** WebGPU needs the OS 26 Metal stack. Safari 26.0.1 on macOS 15 has no `navigator.gpu`; WebKit engineers confirmed it requires macOS Tahoe, iOS 26 or visionOS 26 [31]. On iOS the Safari and OS versions are tied, so the gate is **iOS ≥ 26**.
- **Hardware floor [V]/[U].** iOS 26's floor is the A13 (iPhone 11 / SE 2), which is the Apple6 GPU family [36]. WebKit merges the Apple4–Apple7 capability tables, so every iOS 26 device should expose an adapter [S][16]. The iOS 27 device list is **[U]**: leaks and reports conflict.
- **Default-on flag [S].** `WebGPUEnabled` defaults to `true` under `ENABLE(WEBGPU_BY_DEFAULT)` and has `disableInLockdownMode: true` [21]. Lockdown Mode users therefore get `navigator.gpu === undefined` and need the fallback path.
- **Workers [S].** `GPU.idl` is `Exposed=(Window,Worker)`. `NavigatorGPU` mixes into `WorkerNavigator`, and `WorkerNavigator::gpu()` builds its GPU via `graphicsClient()->createGPUForWebGPU()` [20]. BCD lists `api.WorkerNavigator.gpu` and `OffscreenCanvas.getContext("webgpu")` at Safari 26 [35].
- **PWA / standalone [U].** No WebGPU-specific code path for Home Screen web apps was found. Since iOS 26, every Home Screen site opens as a web app by default and uses the same WebKit [7]. Expect parity, but feature-detect on every start.
- **Detection.** `navigator.gpu` existing is not enough. Treat `requestAdapter()` returning `null` as "no WebGPU".

## 2. Limits on recent iPhones (A15–A19)

A15/A16 are Apple8, A17 Pro/A18 are Apple9, and A19 is Apple10 [36]. All of them support `MTLGPUFamilyApple7`, so they share **one limits table** in WebKit [16]. Only the buffer-size limits (which depend on RAM) and `float32-filterable` differ between them.

| Limit | Spec default (what you get if you don't ask) | iPhone adapter (Safari 26.0 → 27.0) |
|---|---|---|
| `maxBufferSize` | 268,435,456 (256 MiB) | 26.0–26.1: `max(256 MiB, min(INT_MAX, maxBufferLength/10))`, which is **256 MiB** on ≤ 8 GB phones. On 12 GB phones it may be slightly higher (about 300 MiB if `maxBufferLength` is about 3 GB) **[U]**. 26.2+: `max(256 MiB, min(1 GiB, maxBufferLength/3))`. Both are rounded down to a multiple of 4 [S] |
| `maxStorageBufferBindingSize` | 134,217,728 (128 MiB) | **= maxBufferSize** [S] |
| `maxUniformBufferBindingSize` | 65,536 | = maxBufferSize [S] |
| `maxComputeWorkgroupStorageSize` | 16,384 | **32,768** (the Metal hardware maximum is also 32 KB) |
| `maxComputeInvocationsPerWorkgroup` | 256 | **1024** |
| `maxComputeWorkgroupSizeX/Y/Z` | 256 / 256 / 64 | **1024 / 1024 / 1024** (the product is still ≤ 1024) |
| `maxComputeWorkgroupsPerDimension` | 65,535 | 65,535 |
| `maxStorageBuffersPerShaderStage` | 8 | **44** (11 groups × 4) |
| `maxStorageTexturesPerShaderStage` | 4 | **44** |
| `maxSampledTexturesPerShaderStage` | 16 | **44** |
| `maxSamplersPerShaderStage` | 16 | 22 |
| `maxUniformBuffersPerShaderStage` | 12 | 44 |
| `maxBindGroups` | 4 | **11** (capped below 30 because of a CTS issue) |
| `maxBindingsPerBindGroup` | 1000 | 65,535 on Apple families (the Mac2 table uses 1000) |
| `maxTextureDimension2D` / `1D` / `3D` | 8192 / 8192 / 2048 | **16384** / 16384 / 2048 |
| `maxTextureArrayLayers` | 256 | **2048** |
| `minStorageBufferOffsetAlignment` / `minUniformBufferOffsetAlignment` | 256 / 256 | **32 / 32** (must be requested to use) |
| `maxStorageBuffersInFragmentStage` and related | 8 / 4 | 44. These were exposed to JS only from 27.0 [5] |

Sources: [16] for the per-family tables and `maxBufferSize()` on all branches, and [36] for the hardware.

**Concrete `maxBufferSize` values on Safari 26.2+ [U].** The formula comes from source, but WebKit's input is `MTLDevice.maxBufferLength`, which Apple does not publish for iOS. Older measurements put it at about 23–25% of RAM (for example, iPhone 12 Pro with 6 GB reported 1433 MB [37]). That gives these expectations:

- 6 GB device: about 450–480 MiB.
- 8 GB A17 Pro/A18: about 0.6–0.7 GiB.
- 12 GB A19 Pro: 1 GiB (the cap).

Read the value at runtime; never hard-code it. In WebKit's code the `/3` reflects the intent to "allow 3 max-size buffers".

**Requesting limits [S].** `Adapter::requestDevice` uses `defaultLimits()` unless `requiredLimits` is given, and then rejects anything above the adapter's limits [28]. You must therefore pass every limit you need.

- In Safari ≤ 27.0, `undefined` values in `requiredLimits` are not handled; BCD lists `requestDevice.undefined_limits` as unsupported [35]. Copy only numeric values.
- An adapter can create **one** device. A second `requestDevice` call invalidates the adapter [28]. After a loss, call `requestAdapter()` again.
- `adapter.info` reports only `"apple"` for `vendor`, `architecture`, `device` and `description` [S][20]. Use features and limits to infer the device tier:
  - `float32-filterable` present on an iPhone means A17 Pro or newer.
  - `maxBufferSize` is a usable proxy for RAM.

## 3. Optional features and WGSL extensions

### Device features

| Feature | Safari availability | Notes |
|---|---|---|
| `shader-f16` | 26.0+ on all devices [S][35]. Apple says it is supported on all Apple devices but still optional, so check for it [38]. | The main win is register pressure and bandwidth. **[U]** Don't expect 2× ALU throughput on A15+. Never use f16 for world coordinates or depth at backyard scale. |
| `float32-filterable` | Exposed when `MTLDevice.supports32BitFloatFiltering` is true [S][16]. Per the Metal tables, R32Float/RGBA32Float "Filter" is supported on **Apple9/Apple10**; on Apple7/8 only some iPadOS GPUs support it [36]. | So A17 Pro/A18/A19 have it and A15/A16 iPhones do not. BCD's "No" is wrong for real hardware (its collector probably ran on VMs). **[U]** Verify on a device. Fallback: manual bilinear with `textureLoad`, or `rgba16float`. |
| `subgroups` | **Not in 26.x, 27.0 or the 27.2 beta** [5][6]. "Added support for subgroups in WebGPU" appears in STP 249 (2026-07-29) [10]. On main it is gated on `MTLGPUFamilyMetal3`, and `subgroupMin/MaxSize` = the SIMD width (32 on Apple GPUs) [S]. | Write subgroup variants of reductions and prefix-scans only behind `adapter.features.has('subgroups')`. **[U]** Likely to ship in a 27.x or 28 release. |
| `timestamp-query` | Exposed if Metal reports a timestamp counter set with stage-boundary sampling [S][16]. BCD lists Safari 26 [35]. Presence on each specific A-series iPhone is **[U]**. | Pass-level timestamps only. A WebKit engineer (bug 311598, comment 7) says timestamp queries require waiting on an `MTLSharedEvent`, which is heavyweight [29]. Use them in profiling builds only. |
| `bgra8unorm-storage` | 26.0+ [S]. | Write-only storage. |
| `texture-formats-tier1` | 26.2+ [S][35]. | Adds r8/rg8/r16/rg16/rgba16 unorm/snorm, rg11b10 and others as storage and render formats. |
| `texture-formats-tier2` | **27.0+** (`safari-7625-branch` on ARM64) [S]. BCD still says "No" and the Safari 27 release notes don't list it, so this rests on source alone. | Adds `read_write` for rgba8unorm, rgba16float, rgba32float, r8unorm, r16float and others. Always gate on `adapter.features.has('texture-formats-tier2')`. |
| `clip-distances`, `primitive-index` | 27.0 [5][S]. STP 253 fixed `primitive-index` being reported on GPUs that lack it [14]. | Not needed for compute. |
| `dual-source-blending`, `texture-compression-bc` | BC on iOS only on some iPadOS GPUs [36]. | Not relevant. |

### Storage-texture formats usable in compute [S][19]

| Format | 26.0–26.6 | 27.0+ |
|---|---|---|
| `r32float`, `r32uint`, `r32sint` | **read / write / read_write** | Same |
| `rgba16float`, `rgba32float`, `rgba8unorm/uint/sint`, `rgba16uint/sint`, `rgba32uint/sint` | read-only or write-only | **+ read_write** with tier2 |
| `rg32float/uint/sint`, `rgba8snorm` | read-only or write-only | Same |
| `r8unorm`, `r16float`, `r16uint` and others | Storage needs tier1 (26.2+) | + read_write with tier2 |
| `bgra8unorm` | write-only with `bgra8unorm-storage` | Same |

### WGSL language extensions (`navigator.gpu.wgslLanguageFeatures`) [S][18]

| Extension | 26.0 | 27.0 | Main / STP |
|---|---|---|---|
| `readonly_and_readwrite_storage_textures` | Yes | Yes | Yes |
| `packed_4x8_integer_dot_product` | Yes | Yes | Yes |
| `pointer_composite_access` | Yes | Yes | Yes |
| `unrestricted_pointer_parameters` | Yes | Yes | Yes |
| `texture_formats_tier1` | No | Yes | Yes |
| `subgroup_id`, `subgroup_uniformity` | No | No | Yes |

MDN BCD currently marks the first four as "No" for Safari, which disagrees with WebKit source on every release branch since 7622. Always test with `wgslLanguageFeatures.has(...)` at runtime.

### Atomics and barriers

- **No f32 atomics [V].** WGSL `atomic<T>` allows only `i32` and `u32` [39]. Metal has float atomics on Apple7+ [36], but WebGPU does not expose them. Workarounds:
  1. Fixed-point `atomicAdd` on `i32`. Use this for TSDF/voxel weights and accumulators.
  2. For non-negative floats, `atomicMin`/`atomicMax` on `bitcast<u32>(x)` is order-preserving. Use this for z-buffer style depth min when splatting points.
  3. A `atomicCompareExchangeWeak` CAS loop for rare float adds.
  4. Workgroup-local reduction in `var<workgroup>`, then one atomic per workgroup.
- **Barrier lowering [S][25].** `workgroupBarrier()` becomes `threadgroup_barrier(mem_flags::mem_threadgroup)`, `storageBarrier()` becomes `threadgroup_barrier(mem_flags::mem_device)`, and `textureBarrier()` becomes `threadgroup_barrier(mem_flags::mem_texture)`.
  - All of these are **workgroup-scope only**. There is no device-wide sync inside a dispatch; use separate dispatches for that.
  - Dispatches in the same pass are serialized with Metal's automatic hazard tracking (default serial compute encoder) [S].
- **Zero-init cost [S].** Workgroup memory is zero-initialized at shader entry, surrounded by threadgroup barriers [25]. Size `var<workgroup>` arrays deliberately.

## 4. Known quirks for compute-heavy workloads

### 4.1 GPU timeouts, the watchdog and command-buffer failures

**Documented duration: none [U].** Apple does not publish a GPU watchdog value for iOS, and none of the WebKit bugs reviewed cite one.

**What WebKit does when a command buffer fails [S][17]** (`Queue.mm`, compared on `safari-7622` through `safari-7625` and main). The logic is the same on every branch **except one difference on `safari-7625` (27.0)**, called out in the table:

| Metal outcome | WebKit reaction |
|---|---|
| `MTLCommandBufferErrorNotPermitted` (GPU use while backgrounded) | **Device not lost**. The work is silently dropped. |
| Underlying IOGPU code `0x5` (InnocentVictim, "discarded, victim of GPU error/recovery") or `0x10a` | **Device not lost**. The work is silently dropped. |
| IOGPU codes 9 (InvalidResource), 10 (InvalidInput), 11 (PageFault), 16 (ProtectionViolation), 17 (StackOverflow) | **The GPU process deliberately crashes** (`RELEASE_ASSERT_NOT_REACHED`). Every WebGPU device in that WebContent process is lost. |
| **IOGPU code 8 (OutOfMemory)** | **27.0 (`safari-7625`): the GPU process deliberately crashes**, as in the row above. 26.x and main: falls through to the next row. |
| Any other underlying error (timeouts, hangs, ImpactingInteractivity and so on; the numeric codes for these are **[U]**) | `device.lost` resolves with reason `"unknown"` and an empty message. |

Consequence for 27.0: overcommitting GPU memory during execution does not just lose the device. It **kills the GPU-process connection for the whole tab** (§4.3), so `requestAdapter()` in that global then returns `null`.

Other behavior to know:

- **`onSubmittedWorkDone` is not a success signal.** It is called with *Success* for every completed command buffer, whatever its error status [S][17].
- **Command-buffer overflow is reported as "destroyed".** With more than 1000 created-but-uncommitted command buffers, the device is lost with reason `"destroyed"` [S][17]. Do not assume `"destroyed"` means you called `destroy()`; track that yourself.
- **Bug 311598 (filed 2026-04-06, iOS 26, still NEW) [V][29]: stalls with several command buffers in flight.** A llama.cpp WebGPU backend running 400–600 compute passes per GPU/CPU sync, batched 16–32 passes per command buffer, saw queue waits stall and time out on iOS, with *no* device-lost or error.
  - Even two command buffers in flight were enough to trigger it. It persisted on 26.3.1 and was "more stable" on 26.4.
  - Attaching Safari Web Inspector made the problem disappear, so reproduce with the inspector detached.
  - Apple's guidance (comment 2): "1 command buffer or at the very least each submitted command buffer should take several milliseconds of work". WebKit **does not coalesce** command buffers (comment 2) or passes (comment 5).
  - Apple's guidance (comment 3): normally you only need to split at a GPU/CPU sync point.
  - Apple's guidance (comment 5): switch compute pipeline state inside a pass rather than ending and starting passes.
  - The reporter's workarounds: keeping one command buffer in flight worked (comment 4). Putting 64 compute passes per command buffer also stabilized it, and then multiple command buffers in flight appeared stable (comment 6).
- **Bug 290620 [V][30]: long-running shader.** A long fragment shader caused context loss on an iPhone 13 mini and an iPad Pro 2020 (WebGL over ANGLE/Metal), with `kIOGPUCommandBufferCallbackErrorInnocentVictim`. A WebKit engineer notes that iOS 26 improved infinite-loop mitigation.
- **Driver hang on main [S].** A blit-channel deadlock was triggered by `writeBuffer` staging interleaved with compute: the driver reported `Hang` and losses spread to neighbouring buffers as innocent victims. Main now routes staged copies through compute [17]. Shipping 26.x/27.0 may still hit this **[U]**, so keep `writeBuffer` traffic out of heavy compute submits.

**Engineering estimate [U].** Metal reports `Timeout` for command buffers that run for multiple seconds. When the compositor is starved, iOS can also cancel foreground work much sooner (`ImpactingInteractivity`). Treat **about 100 ms of continuous GPU time per command buffer as a hard ceiling** and 2–16 ms as the normal operating range. §5 covers the scheduler.

### 4.2 Memory

- **GPU memory counts against the tab's jetsam budget [S][27].** Metal buffers and textures are re-attributed to the WebContent process (`setOwnerWithIdentity`, and `IOSurfaceSetOwnershipIdentity` with `kIOSurfaceMemoryLedgerTagGraphics`).
- **The limits [V][32][33].** WebContent on iPhone has a soft limit of "around 1.5GB" (WebKit engineer, 2024) [32]. A 2026 report on an iPhone 17 Pro (12 GB) running iOS 26.3–26.4.1 shows `ActiveHard 2048 MB (fatal)` for WebContent; iPads get more [33].
- **Over-budget behavior.** The tab is killed and reloaded, with **no `device.lost`**.
- **GPU OOM at execution time on 27.0 [S][17].** It crashes the GPU process (§4.1), which is worse than a `device.lost`. Wrap large `createBuffer`/`createTexture` calls in `pushErrorScope('out-of-memory')` / `popErrorScope()` so allocation failures are caught early, and keep headroom below the budget.
- **Budget [U].** On iPhone, keep the total WebContent footprint (JS heap + Wasm memory + GPU resources) **≤ about 1.2 GB** in total, and about 0.8 GB on 6 GB devices. Call `destroy()` on buffers and textures explicitly rather than waiting for GC.
- **The Wasm heap is fragile on iOS [V][34].** Pages have managed about 1 GB on first load but only 300–500 MB after a reload. Large `memory.grow` requests can kill the tab instead of failing cleanly.
- **Fixed on main [S].** About 37,000 render encoders in one command buffer caused `OutOfMemory` and device loss. This is irrelevant to compute.

### 4.3 Backgrounding, GPU-process crashes and device loss

- **Backgrounding.** iOS forbids GPU work in the background. Such command buffers fail with `NotPermitted`, which WebKit treats as non-fatal: **no `device.lost`, results silently missing, `onSubmittedWorkDone` still resolves** [S][17].
- **Is the device lost after a long background stay? [U]** Usually not. The GPU process may be suspended or jetsammed, which leads to the next point.
- **GPU-process crash or connection close [S][24].**
  - `RemoteGPUProxy` sets `m_lost = true` permanently, and `requestAdapter()` then returns `null` *for that global* (`Navigator::gpu()` is created once per Navigator [20]).
  - Recovery therefore needs a **fresh global**: terminate and respawn the compute worker, since a new `WorkerNavigator` gets a new GPU connection. If that also fails, reload the page.
  - **[U]** That a respawned worker reconnects successfully is inferred from source; test it.
- **`device.lost` reason [S].** Expect `"unknown"`, or `"destroyed"` in the overflow case above. `message` is empty in WebCore.
- **Error events [V][5], [S][40].** Safari 27.0 "Fixed `GPUDevice.onuncapturederror` event handler attribute not working" [5].
  - In 26.0–26.6 the IDL has **no `onuncapturederror` attribute at all**.
  - In 26.x, `addEventListener('uncapturederror', fn)` arms a **one-shot** native callback: `Device::m_uncapturedErrorCallback` is nulled after it fires. Only the **first** uncaptured error after each `addEventListener` call is delivered.
  - On 26.x, call `device.addEventListener('uncapturederror', fn)` again inside `fn` to re-arm. WebKit's override re-arms even when the listener is already registered.
  - 27.0 re-arms automatically (`listenForUncapturedErrors`).
  - Use `pushErrorScope`/`popErrorScope` for anything you must not miss. BCD lists `uncapturederror_event` as Safari 27 only [35].

### 4.4 `mapAsync` and readback

- WebGPU runs in the GPU process.
- **`mapAsync`** is an async IPC round trip, resolved when the GPU process has mapped the buffer [S][23].
- **`getMappedRange()`** is a **synchronous IPC** that returns a copy of the range: `sendSync(GetMappedRange)` ships a `Vector<uint8_t>` [23]. Each call costs a blocking round trip plus a memcpy, so:
  - call it **once per map** over exactly the range you need;
  - prefer doing it in a worker, since it blocks that thread.
- **Uploads.** `writeBuffer` and mapped writes above 16 MiB go through shared memory; smaller ones are copied inline [23].
- **Latency [U].** No published figures. Expect "end of GPU work + about 1–3 ms IPC" for small buffers (≤ 1 MB). Measure on the device.
- **Design rule.** Read back only small results (poses, residuals, inlier counts, histograms, compacted keypoints), at most once per command-buffer batch, and keep dense data on the GPU.

### 4.5 Getting camera frames into the GPU

| Path | 26.x / 27.0 behavior | Use? |
|---|---|---|
| `device.importExternalTexture({source: VideoFrame or <video>})` | The GPU process wraps the frame's **CVPixelBuffer** with `CVMetalTextureCache`: **zero-copy**, YUV handled by the sampler [S][22]. **[U]** That a camera VideoFrame from MSTP reaches this path without a web-process copy is inferred, not traced end to end; measure the per-frame cost. Available from 26.0 (BCD [35]). Bug fixes: 26.1 (black screen with WebCodecs renderer) [2]; STP 253 (WebM/MSE black frames) [14]. | **Yes, the primary path.** `texture_external` is sampled in a compute shader with `textureSampleBaseClampToEdge` or `textureLoad`, and converted into our own `r16float`/`rgba16float`/`r8unorm` pyramid. |
| `queue.copyExternalImageToTexture({source})` | Sources: ImageBitmap, ImageData, `<img>`, `<video>`, VideoFrame, `<canvas>`, OffscreenCanvas [S][22]. **On 26.x/27.0 pixels are read into the web process** (`getPixelBuffer`, or the CGImage path for video frames) and re-uploaded with `writeTexture`. That is CPU work plus IPC [S]. A GPU-resident path exists only on main/STP: STP 253 fixed canvas uploads having "very poor performance", and STP 254 preserves alpha [14][15]. | Only for full-resolution stills (for example a `takePhoto()` blob turned into an ImageBitmap), not per-frame video. BCD says `<img>`/ImageData sources are unsupported, but the IDL includes them; test. |
| Camera into a worker | `MediaStreamTrack` is `Exposed=(Window,DedicatedWorker)`, so it is transferable. `MediaStreamTrackProcessor` is **DedicatedWorker-only** and on by default on Cocoa (Safari 18+) [S][35]. | Use getUserMedia on the main thread, `postMessage(track, [track])` to the worker, then MSTP → VideoFrame → `importExternalTexture`, all in the worker. Call `frame.close()` promptly. |

### 4.6 Shader compilation and codegen overheads

- **Pipeline creation blocks the GPU process [S][26].** On 26.x/27.0, `createComputePipelineAsync` compiles **synchronously inside the GPU process**; only the callback is deferred. A compile therefore blocks that device's GPU-process work queue.
  - Main has a real async path ("asynchronousIfPossible").
  - STP 253 fixed "asynchronous pipeline creation could stall rendering". STP 254 fixed `createComputePipelineAsync()` and `createRenderPipelineAsync()` "never settling **when the device was lost or destroyed**" [14][15] [V]. Whether shipping 26.x/27.0 has the same hang is **[U]**.
  - Mitigations:
    - **Create every pipeline at startup**, before capture begins.
    - Race each async creation against `device.lost` and a timeout, falling back to the sync `createComputePipeline` when the device is still alive.
    - Never `await` a pipeline promise during loss recovery.
- **Hidden per-iteration cost in loops [S][25].** On 26.x/27.0, every WGSL loop gets a `volatile uint32_t` forward-progress counter that is incremented each iteration. Main uses a cheaper volatile read on Apple GPU families below 9.
  - Mitigation: hand-unroll hot fixed-size inner loops (patch windows, descriptor bits, small linear solves).
- **Bounds checks [S].** Non-constant array indexing gets bounds checks (WGSL `BoundsCheck` pass).
- **Recent WGSL miscompiles (fixed in STP, possibly present in shipping) [V][12][14]:**
  - swizzling a packed `vec3`;
  - inlining 3 or more calls on a large struct;
  - an array constructor reached through a type alias.

  Mitigations: avoid `vec3` in storage (use `vec4`), avoid large structs passed by value, and run a GPU-vs-CPU self-test of the kernels at startup.
- **Indirect dispatch overhead [S].** `dispatchWorkgroupsIndirect` adds a small validation dispatch [S].

## 5. Splitting long compute work, and detecting and recovering from loss

### Scheduling rules

The numbers are engineering estimates **[U]**, based on Apple's guidance in [29].

1. **Granularity.** Aim for **2–8 ms of GPU time per command buffer** while the camera preview is animating, and **≤ 30 ms** in "processing" mode with a static UI. Never plan more than about 100 ms. Inside a command buffer, put many dispatches in **as few compute passes as possible**, switching pipelines within a pass.
2. **Queue depth.** Keep **at most 1 heavy command buffer in flight** on iOS (`await queue.onSubmittedWorkDone()` before submitting the next), and encode the next batch on the CPU while you wait. Raise to 2 only if a device test passes.
3. **Tiling.**
   - Dense MVS / PatchMatch: one propagation iteration per tile set per command buffer, using 256×256 to 512×512 pixel tiles × a depth-hypothesis slab.
   - Feature extraction: per pyramid level.
   - Matching: per image-pair batch.
   - Bundle adjustment: Jacobian/Hessian assembly in blocks of N residuals.
4. **Adaptive controller.** Time each submit with `performance.now()` around `onSubmittedWorkDone`. Use timestamp queries only in dev builds. Shrink or grow the tile count (multiplicative decrease, additive increase) to stay inside the budget. iPhones throttle thermally under sustained load, so the budget must adapt continuously.
5. **Epoch sentinel.** End every command buffer with a 1-thread dispatch that writes `epoch++` into a small status buffer, and read it with the batch's results. If the epoch did not advance, the command buffer was **silently discarded** (`NotPermitted` or InnocentVictim, see §4.1/§4.3): discard the outputs and resubmit.
6. **Visibility.** On `visibilitychange` (hidden) or `pagehide`:
   - stop submitting;
   - `await onSubmittedWorkDone()`;
   - checkpoint compact state (poses, tracks, completed depth tiles) to **OPFS**.

   On visible, run a tiny probe dispatch with readback before resuming.
7. **Memory watermark.** Keep an allocator ledger and stay below the budget in §4.2. Prefer fewer, larger buffers sub-allocated by offset (`minStorageBufferOffsetAlignment` = 32 when requested).

### Recovery state machine

1. Always attach `device.lost.then(info => …)` and an `uncapturederror` listener (not the attribute; see §4.3). On 26.x, re-arm the listener inside its own handler, because it is one-shot there.
2. On loss:
   1. Mark all GPU objects invalid.
   2. `requestAdapter()` again; the old adapter cannot make a second device [28].
   3. Re-request the device with the same limits and features.
   4. Rebuild pipelines from cached WGSL.
   5. Re-upload from the OPFS checkpoint.
3. If `requestAdapter()` returns `null`, the GPU-process connection is dead for this global [24]. **Terminate the compute worker and spawn a new one.** If that also yields `null`, persist and `location.reload()`.
4. Cap retries (for example 3 within 60 s). Then drop to a lower tier (smaller tiles, fewer views, no dense MVS) or to the Wasm path.
5. A jetsam kill gives no signal at all. Resume from OPFS on the next load, using a "session in progress" marker.

### Device request snippet (runs in the worker)

```js
const adapter = await navigator.gpu?.requestAdapter();
if (!adapter) return fallback();
const want = ['shader-f16', 'float32-filterable', 'subgroups', 'texture-formats-tier1',
              'texture-formats-tier2', 'bgra8unorm-storage'];      // add 'timestamp-query' only in profiling builds
const requiredFeatures = want.filter(f => adapter.features.has(f));
const requiredLimits = {};
for (const k of ['maxBufferSize', 'maxStorageBufferBindingSize', 'maxComputeWorkgroupStorageSize',
                 'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX', 'maxComputeWorkgroupSizeY',
                 'maxComputeWorkgroupSizeZ', 'maxStorageBuffersPerShaderStage', 'maxStorageTexturesPerShaderStage',
                 'maxSampledTexturesPerShaderStage', 'maxBindGroups', 'maxTextureDimension2D',
                 'maxTextureArrayLayers', 'minStorageBufferOffsetAlignment', 'minUniformBufferOffsetAlignment'])
  if (typeof adapter.limits[k] === 'number') requiredLimits[k] = adapter.limits[k];   // never pass undefined (Safari ≤27)
const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
const wgsl = navigator.gpu.wgslLanguageFeatures;   // check readonly_and_readwrite_storage_textures, packed_4x8_integer_dot_product, ...
```

## 6. Fallbacks

### WebGL2 as a compute-like fallback

WebGL2 is available on all supported iOS versions, implemented through ANGLE on Metal. It has **no compute shaders**, no SSBOs, no atomics, no shared memory and no scatter except through point-sprite blending, and `readPixels` stalls.

- **Works with fragment-shader GPGPU on float render targets:** image pyramids, gradient/Harris/FAST scores, ZNCC cost volumes, plane-sweep stereo, and multi-pass reductions.
- **Effectively not viable:** PatchMatch with random propagation at scale, GPU bundle adjustment, and TSDF fusion (that one needs point-splat blending tricks).
- **Who would use it:** since every iOS 26 device has WebGPU, the only audience is iOS ≤ 18 and Lockdown Mode users.
- **Recommendation:** do not build a WebGL2 SfM backend. Use **Wasm SIMD + threads** for the CPU pipeline, and WebGL2 for preview and pyramids only.

### WebAssembly SIMD and threads on iOS Safari [35]

- **SIMD and threads [V].** Fixed-width SIMD128 since Safari **16.4**. Threads, atomics and `SharedArrayBuffer` since **15.2**, and only when the page is `crossOriginIsolated`:
  - `Cross-Origin-Opener-Policy: same-origin`;
  - `Cross-Origin-Embedder-Policy: require-corp`. Safari does **not** support `credentialless` [35], so every cross-origin subresource needs CORP/CORS headers.
- **Recent fixes.** Safari 27.0 fixed `SharedArrayBuffer` cloning and agent-cluster assignment, and cross-origin-isolation enforcement for workers [5]. STP 252 fixed `worker.terminate()` not ending workers blocked in `Atomics.wait` [13].
- **Not shipping yet [V].** Relaxed SIMD (STP 250) [11] and memory64 / multi-memory (STP 252) [13] are STP only; BCD lists them as "preview". JSPI shipped in **27.0** [5][9].
- **SIMD correctness on 26.x [V][5].** Safari 27.0 "Fixed a regression where `RegisterSet::normalizeWidths()` lost vector-width information, causing v128 argument corruption in WebAssembly SIMD thunks." Which 26.x builds are affected is **[U]**. Avoid passing `v128` values across function-call boundaries through imports or `call_indirect` in hot code, and include SIMD kernels in the startup self-test.
- **Memory budget.** Declare a modest `maximum` for shared memory (for example 512 MiB–1 GiB) and pre-size it once rather than growing it repeatedly (§4.2, [34]).

## 7. Recommendations for our engine

1. **Gate on iOS 26+ with a WebGPU adapter, and run the whole compute engine in a DedicatedWorker:** device, camera frames via MSTP, OPFS persistence. The main thread does UI and getUserMedia only.
2. **Request the limits explicitly** (snippet in §5). Design for **32 KiB workgroup memory and 1024 invocations**, with workgroup sizes that are multiples of 32 (64–256 typical). Plan for **256 MiB maximum per buffer/binding** (Safari 26.0–26.1) and scale up only when `maxBufferSize` allows.
3. **Ingest frames with `importExternalTexture(VideoFrame)` and one compute conversion pass** into an `r16float` (or `r8unorm`) luma pyramid. Avoid `copyExternalImageToTexture` for video on 26.x/27.0.
4. **Assume no subgroups and no f32 atomics.** Write workgroup-memory reductions and scans, use fixed-point `i32` atomics for accumulation and `bitcast<u32>` `atomicMin` for depth. Add subgroup fast paths behind feature detection.
5. **Use f16 for intensities, descriptors and cost volumes, and f32 for geometry.** Large-scene coordinates should be relative to a local origin per keyframe or tile, in f32.
6. **Storage textures.** Rely on `r32float`/`r32uint` `read_write` (26.0+). Use `rgba16float` `read_write` only when `texture-formats-tier2` is present (27.0+); otherwise use ping-pong textures or storage buffers.
7. **Scheduler.** 2–30 ms command buffers, one in flight, an adaptive tile count, an epoch sentinel per command buffer, and a pause/checkpoint on visibility change (§5).
8. **Resilience.**
   - Handle `device.lost` (re-adapter, then re-device, then rebuild).
   - On a `null` adapter, respawn the worker; then reload with an OPFS resume. On 27.0 a GPU out-of-memory crashes the GPU process, so this path is not rare.
   - Treat `onSubmittedWorkDone` as "finished", not "succeeded".
   - Re-arm `uncapturederror` on 26.x, and use error scopes around large allocations.
9. **Memory.** Track all allocations. Keep the WebContent total around 1.2 GB or less (0.8 GB on 6 GB phones). Destroy explicitly. Stream depth maps and fused geometry to OPFS instead of keeping everything resident.
10. **Startup self-test** (about 200 ms): compile all pipelines up front (pipeline creation blocks the GPU queue on 26.x/27.0), then run each kernel on a tiny fixed input and compare with a JS reference. This catches WGSL miscompiles on a given OS build; fall back to an alternative kernel variant if a check fails.
11. **WGSL style.** Use `vec4` storage layouts and no `vec3` in buffers. Hand-unroll fixed small loops, because of the per-iteration volatile counter. Avoid passing large structs by value.
12. **Profiling.** Use `timestamp-query` and Web Inspector's WebGPU Canvas tab (STP 251+) in dev builds only.

## 8. Open items to verify on a device (test matrix: A15, A16, A17 Pro, A18, A19 × iOS 26.0, 26.4, 27.0)

- Actual `maxBufferSize` per device, and the `features` set (`float32-filterable`, `timestamp-query`).
- How long a single dispatch can run before a timeout or `ImpactingInteractivity`, with a static UI and with an animating preview.
- Whether the device is lost after 30 s, 5 min and 30 min in the background, or after a screen lock.
- Whether a GPU-process kill can be recovered by respawning the worker.
- `mapAsync` + `getMappedRange` latency for 4 KB, 1 MB and 16 MB readbacks.
- `importExternalTexture(VideoFrame)` cost per frame at 1920×1440, and whether MSTP delivers full sensor resolution.
- Behavior in Home Screen (standalone) mode, which should be identical.

---

## Citations

1. Apple, *Safari 26.0 Release Notes*, "Added support for WebGPU. (145801580)", plus Web Apps entries — https://developer.apple.com/documentation/safari-release-notes/safari-26-release-notes
2. Apple, *Safari 26.1 Release Notes*, WebGPU fixes (`copyExternalImageToTexture` with SVG, WebCodecs black screen, video textures) — https://developer.apple.com/documentation/safari-release-notes/safari-26_1-release-notes
3. Apple, *Safari 26.2 Release Notes*, `GPUTexture` as attachments; WebGPU for WebXR — https://developer.apple.com/documentation/safari-release-notes/safari-26_2-release-notes
4. Apple, *Safari 26.4 Release Notes*, PNG pixel-format fix; DeviceMotion/Orientation secure-context alignment — https://developer.apple.com/documentation/safari-release-notes/safari-26_4-release-notes
5. Apple, *Safari 27 Release Notes*, WebGPU (`clip_distances`, `onuncapturederror` fix, restored fragment/vertex storage limits, WGSL validation fixes), SharedArrayBuffer and worker-isolation fixes — https://developer.apple.com/documentation/safari-release-notes/safari-27-release-notes
6. Apple, *Safari 27.2 Beta Release Notes*, WebGPU `executeBundles` indirect-draw fix only — https://developer.apple.com/documentation/safari-release-notes/safari-27_2-release-notes
7. WebKit blog, *News from WWDC25: WebKit in Safari 26 beta* (2025-06-09) — https://webkit.org/blog/16993/news-from-wwdc25-web-technology-coming-this-fall-in-safari-26-beta/
8. WebKit blog, *News from WWDC26: WebKit in Safari 27 beta* — https://webkit.org/blog/17967/news-from-wwdc26-webkit-in-safari-27-beta/
9. WebKit blog, *WebKit Features for Safari 27.0* (2026-09-17) — https://webkit.org/blog/18325/webkit-features-for-safari-27-0/
10. WebKit blog, *Release Notes for STP 249* (2026-07-29), "Added support for subgroups in WebGPU." — https://webkit.org/blog/18182/release-notes-for-safari-technology-preview-249/
11. WebKit blog, *STP 250* (2026-08-13), Wasm relaxed SIMD — https://webkit.org/blog/18191/release-notes-for-safari-technology-preview-250/
12. WebKit blog, *STP 251* (2026-08-26), WGSL packed-`vec3` swizzle miscompile fix — https://webkit.org/blog/18194/release-notes-for-safari-technology-preview-251/
13. WebKit blog, *STP 252* (2026-09-11), Wasm memory64 + multi-memory; `Atomics.wait` termination fix — https://webkit.org/blog/18304/release-notes-for-safari-technology-preview-252/
14. WebKit blog, *STP 253* (2026-09-23), canvas upload performance, async pipeline stall, WGSL miscompiles, render-bundle device loss — https://webkit.org/blog/18357/release-notes-for-safari-technology-preview-253/
15. WebKit blog, *STP 254* (2026-10-08), `create*PipelineAsync` never settling, `copyExternalImageToTexture` alpha — https://webkit.org/blog/18420/release-notes-for-safari-technology-preview-254/
16. WebKit source, `Source/WebGPU/WebGPU/HardwareCapabilities.mm` (limits, `maxBufferSize()`, `baseFeatures()`), compared across `safari-7621`…`safari-7625` branches and main — https://github.com/WebKit/WebKit/blob/safari-7625-branch/Source/WebGPU/WebGPU/HardwareCapabilities.mm
17. WebKit source, `Source/WebGPU/WebGPU/Queue.mm` (command-buffer completion handling, `NotPermitted`, IOGPU codes, 1000-command-buffer cap, blit-hang comment on main) — https://github.com/WebKit/WebKit/blob/main/Source/WebGPU/WebGPU/Queue.mm
18. WebKit source, `Source/WebCore/Modules/WebGPU/WGSLLanguageFeatures.cpp` (`safari-7622`, `safari-7625`, main) — https://github.com/WebKit/WebKit/blob/safari-7622-branch/Source/WebCore/Modules/WebGPU/WGSLLanguageFeatures.cpp
19. WebKit source, `Texture::hasStorageBindingCapability` in `Source/WebGPU/WebGPU/Texture.mm` (`safari-7622` vs main) — https://github.com/WebKit/WebKit/blob/main/Source/WebGPU/WebGPU/Texture.mm
20. WebKit source, `GPU.idl` (`Exposed=(Window,Worker)`), `NavigatorGPU.idl`, `page/WorkerNavigator.cpp`, `page/Navigator.cpp`, `GPUAdapterInfo.h` — https://github.com/WebKit/WebKit/tree/main/Source/WebCore/Modules/WebGPU
21. WebKit source, `Source/WTF/Scripts/Preferences/UnifiedWebPreferences.yaml` (`WebGPUEnabled`, `disableInLockdownMode`, `MediaStreamTrackProcessingEnabled`) — https://github.com/WebKit/WebKit/blob/main/Source/WTF/Scripts/Preferences/UnifiedWebPreferences.yaml
22. WebKit source, `GPUQueue.cpp` (`copyExternalImageToTexture`, `safari-7622`/`safari-7625` vs main), `GPUImageCopyExternalImage.idl`, `GPUExternalTextureDescriptor.idl`, `Source/WebGPU/WebGPU/ExternalTexture.mm`; `Modules/mediastream/MediaStreamTrackProcessor.idl` — https://github.com/WebKit/WebKit/blob/safari-7625-branch/Source/WebCore/Modules/WebGPU/GPUQueue.cpp
23. WebKit source, `Source/WebKit/WebProcess/GPU/graphics/WebGPU/RemoteBufferProxy.cpp/.h` (sync `GetMappedRange`, 16 MiB shared-memory threshold) — https://github.com/WebKit/WebKit/blob/main/Source/WebKit/WebProcess/GPU/graphics/WebGPU/RemoteBufferProxy.cpp
24. WebKit source, `RemoteGPUProxy.cpp` (`abandonGPUProcess`, `m_lost`) — https://github.com/WebKit/WebKit/blob/main/Source/WebKit/WebProcess/GPU/graphics/WebGPU/RemoteGPUProxy.cpp
25. WebKit source, `Source/WebGPU/WGSL/Metal/MetalFunctionWriter.cpp` (forward-progress loop guards, barrier lowering, workgroup zero-init) — https://github.com/WebKit/WebKit/blob/safari-7625-branch/Source/WebGPU/WGSL/Metal/MetalFunctionWriter.cpp
26. WebKit source, `Source/WebGPU/WebGPU/ComputePipeline.mm` (`createComputePipelineAsync`) — https://github.com/WebKit/WebKit/blob/safari-7625-branch/Source/WebGPU/WebGPU/ComputePipeline.mm
27. WebKit source, `Source/WebGPU/WebGPU/Device.mm` / `Buffer.mm` (`setOwnerWithIdentity`) — https://github.com/WebKit/WebKit/blob/main/Source/WebGPU/WebGPU/Device.mm
28. WebKit source, `Source/WebGPU/WebGPU/Adapter.mm` (`requestDevice`: default limits, single device per adapter) — https://github.com/WebKit/WebKit/blob/main/Source/WebGPU/WebGPU/Adapter.mm
29. WebKit Bug 311598, *Limitations on submitted compute WebGPU CommandBuffers on iOS* (2026-04, open; comments by M. Wyrzykowski) — https://bugs.webkit.org/show_bug.cgi?id=311598
30. WebKit Bug 290620, *WebGL context lost …* (long-running shader, InnocentVictim) — https://bugs.webkit.org/show_bug.cgi?id=290620
31. WebKit Bug 299237, *WebGPU: navigator.gpu is not defined: Version 26.0.1 / Sequoia 15.7* — https://bugs.webkit.org/show_bug.cgi?id=299237
32. WebKit Bug 277848, S. Fraser: WebContent on iPhone "soft limit of around 1.5GB" (2024-08-09) — https://bugs.webkit.org/show_bug.cgi?id=277848
33. Apple Developer Forums thread 823061: WebContent `ActiveHard 2048 MB` on iPhone 17 Pro, iOS 26.3–26.4.1 — https://developer.apple.com/forums/thread/823061
34. WebKit Bug 269777, *Only ~300 MB of RAM is reliably available to WebAssembly pages* — https://bugs.webkit.org/show_bug.cgi?id=269777
35. MDN browser-compat-data (main, fetched 2026-10-16): `api/GPUSupportedFeatures.json`, `api/WGSLLanguageFeatures.json`, `api/WorkerNavigator.json`, `api/GPUQueue.json`, `api/GPUDevice.json`, `api/GPUAdapter.json`, `api/OffscreenCanvas.json`, `api/MediaStreamTrackProcessor.json`, `webassembly/*.json`, `http/headers/Cross-Origin-Embedder-Policy.json` — https://github.com/mdn/browser-compat-data
36. Apple, *Metal Feature Set Tables* (2026-05-21): GPU-family mapping (A15/A16 = Apple8, A17 Pro/A18 = Apple9, A19 = Apple10), 32 KB threadgroup memory, 1024 threads, R32Float/RGBA32Float filter support (Apple9+; some Apple7/8 iPadOS GPUs), float atomics (Apple7) — https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
37. gfx-rs/wgpu PR #2502: measured `maxBufferLength` (iPhone 12 Pro 1433 MB, iPhone 8 Plus 747 MB, …) — https://github.com/gfx-rs/wgpu/pull/2502
38. Apple WWDC25 session 236, *Unlock GPU computing with WebGPU* (shader-f16 on all Apple devices, still optional) — https://developer.apple.com/videos/play/wwdc2025/236/
39. W3C, *WebGPU* (limits) and *WGSL* (atomic types: `i32`/`u32` only) — https://www.w3.org/TR/webgpu/#limits , https://www.w3.org/TR/WGSL/#atomic-types
40. WebKit source, `Source/WebCore/Modules/WebGPU/GPUDevice.idl` / `GPUDevice.cpp` (`addEventListener` → one-shot `resolveUncapturedErrorEvent` on `safari-7622`…`safari-7624`; `listenForUncapturedErrors` re-arm and `attribute EventHandler onuncapturederror` on `safari-7625`) and `Source/WebGPU/WebGPU/Device.mm` (`m_uncapturedErrorCallback = nullptr` after firing) — https://github.com/WebKit/WebKit/blob/safari-7622-branch/Source/WebCore/Modules/WebGPU/GPUDevice.cpp

---

## Review notes (skeptical review, 2026-10-16)

I spot-checked the most consequential claims against primary sources: WebKit source on `safari-7622/7623/7624/7625-branch` and `main`, the Apple release-notes JSON for 26.0/26.1/26.2/26.4/27.0/27.2, the WebKit blog for STP 249/253/254, bugs.webkit.org 277848 and 311598, the Apple Developer Forums thread 823061, and MDN BCD `main`.

**Verified, no change needed**

- WebGPU is on by default from 26.0, with release-note entry 145801580. The `WebGPUEnabled` preference has `disableInLockdownMode: true`.
- `maxBufferSize()` formulas on every branch. `maxStorageBufferBindingSize` and `maxUniformBufferBindingSize` are both set to `maxBufferSize`.
- Compute and binding limits: 32 KiB, 1024, 11 bind groups, 44/22, and alignment 32 (min-merge).
- Storage-texture `read_write` rules in `Texture::hasStorageBindingCapability`.
- WGSL language features on 7622, 7625 and main.
- `subgroups`: absent from the 27.0 and 27.2-beta notes and the 7625 source. Present in STP 249 (2026-07-29) and on main, gated on `MTLGPUFamilyMetal3`.
- `Queue.mm` behavior: `NotPermitted` and InnocentVictim are non-fatal, `onSubmittedWorkDone` always reports Success, and the 1000-command-buffer cap is reported as "destroyed".
- `getMappedRange` is a `Synchronous` IPC returning `std::optional<Vector<uint8_t>>`. Writes above 16 MB (`maxCrossProcessResourceCopySize`) use shared memory.
- On 26.x/27.0, `createComputePipelineAsync` compiles synchronously, and `copyExternalImageToTexture` reads back through CGImage/`getPixelBuffer` and uploads with `writeTexture`.
- The per-loop volatile forward-progress counter, and the barrier lowering.
- `requestDevice` uses default limits unless they are requested, and allows one device per adapter.
- Memory limits: S. Fraser's "soft limit of around 1.5GB", and `ActiveHard 2048 MB (fatal)` on an iPhone 17 Pro.
- Wasm: COEP `credentialless` is "No" in BCD, relaxed SIMD and memory64 are "preview", and JSPI is 27.

**Corrected**

1. The WebKit-branch-to-Safari mapping was marked "inferred". It is now **verified** with Apple's release-note build numbers (26.2 = 20623.1.14, 26.4 = 20624.1.16, 27.0 = 20625.1.29, 27.2 beta = 20625.2.8).
2. `Queue.mm` is **not** identical across branches. On `safari-7625` (27.0), **IOGPU code 8 (OutOfMemory) crashes the GPU process**; 26.x and main treat it as a normal `device.lost`. I added this to the summary, §4.1, §4.2 and the recommendations. I also marked the numeric Timeout/Hang code labels **[U]**.
3. `uncapturederror` on 26.x is **one-shot per `addEventListener` call** (`Device::m_uncapturedErrorCallback` is nulled after firing), and the attribute is missing from the 26.x IDL. The report had said `addEventListener` was a full workaround. I added the re-arm guidance to §4.3 and §5, plus citation [40].
4. `texture-formats-tier2` at 27.0 had been attributed to BCD as well as source. **BCD says "No"** and the release notes are silent, so the claim now rests on [S] only.
5. 26.0–26.1 `maxBufferSize` is not a flat 256 MiB. The formula is `max(256 MiB, maxBufferLength/10)`, so 12 GB phones may see slightly more; that value is marked **[U]**.
6. Bug 311598 details were corrected against the bug text: batches of 16–32 passes, persisted on 26.3.1, the Web Inspector masks the issue, Apple's comment 3, and the reporter's workarounds as stated in comments 4 and 6. "Fairly heavyweight" is no longer presented as a quote.
7. STP 254's async-pipeline fix applies to pipelines "never settling **when the device was lost or destroyed**". The earlier "main-only regression" guess was replaced with **[U]** plus recovery guidance.
8. Added the Safari 27.0 Wasm SIMD v128 thunk-corruption fix as a 26.x risk; the affected builds are **[U]**.
9. Marked zero-copy `importExternalTexture` for MSTP camera frames as partly **[U]**: the GPU-side `CVMetalTextureCache` path is verified, but the end-to-end frame path is not.

**Still unverified (no primary source):** the actual `MTLDevice.maxBufferLength` per iPhone, the GPU watchdog duration, behavior of standalone Home Screen apps, and whether a respawned worker reconnects after a GPU-process crash. All of these remain on the device test matrix in §8.
