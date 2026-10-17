/**
 * WebGPU device acquisition shared by every GPU module (features, matching,
 * dense stereo, undistortion). One device per JS realm (main thread or worker).
 *
 * We request the largest adapter limits that matter for compute-heavy image
 * processing, plus optional features when available. Callers must always be
 * prepared for `getGPU()` resolving to null (no WebGPU) and fall back to CPU.
 */

export interface GPUContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  /** Optional features that were successfully enabled on the device. */
  features: ReadonlySet<string>;
  limits: GPUSupportedLimits;
  /** Adapter description for diagnostics. */
  info: { vendor: string; architecture: string; description: string };
  /** True when running on a software rasteriser (SwiftShader etc.) — tests only. */
  isSoftware: boolean;
}

const OPTIONAL_FEATURES: GPUFeatureName[] = [
  'shader-f16',
  'timestamp-query',
  'float32-filterable',
  'subgroups' as GPUFeatureName,
];

const LIMITS_TO_MAXIMISE: (keyof GPUSupportedLimits)[] = [
  'maxStorageBufferBindingSize',
  'maxBufferSize',
  'maxComputeWorkgroupStorageSize',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupSizeY',
  'maxComputeWorkgroupSizeZ',
  'maxBindGroups',
  'maxStorageBuffersPerShaderStage',
  'maxStorageTexturesPerShaderStage',
  'maxSampledTexturesPerShaderStage',
  'maxTextureDimension2D',
  'maxTextureArrayLayers',
  'maxComputeWorkgroupsPerDimension',
  // "Lower is better" limits: requesting the adapter's value is valid.
  'minStorageBufferOffsetAlignment',
  'minUniformBufferOffsetAlignment',
];

let contextPromise: Promise<GPUContext | null> | null = null;
const lostListeners = new Set<(info: GPUDeviceLostInfo) => void>();

/** True if the WebGPU API is exposed in this realm (does not guarantee an adapter). */
export function hasWebGPU(): boolean {
  return typeof navigator !== 'undefined' && !!(navigator as Navigator & { gpu?: GPU }).gpu;
}

/**
 * Returns the shared GPU context, creating it on first call. Resolves to null if
 * WebGPU is unavailable or adapter/device creation fails. After a device loss the
 * next call creates a fresh device.
 */
export function getGPU(): Promise<GPUContext | null> {
  if (!contextPromise) {
    contextPromise = createContext().catch((err) => {
      console.warn('[gpu] WebGPU initialisation failed:', err);
      return null;
    });
    // A null result is not cached forever: allow a retry later (e.g. after a
    // transient failure while the page was backgrounded).
    contextPromise.then((ctx) => {
      if (!ctx) contextPromise = null;
    });
  }
  return contextPromise;
}

/** Register a callback fired when the shared device is lost. Returns an unsubscribe fn. */
export function onDeviceLost(cb: (info: GPUDeviceLostInfo) => void): () => void {
  lostListeners.add(cb);
  return () => lostListeners.delete(cb);
}

async function createContext(): Promise<GPUContext | null> {
  if (!hasWebGPU()) return null;
  const gpu = (navigator as Navigator & { gpu: GPU }).gpu;
  let adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return null;

  const first = adapter;
  const requiredFeatures = OPTIONAL_FEATURES.filter((f) => first.features.has(f));
  const requiredLimits: Record<string, number> = {};
  for (const key of LIMITS_TO_MAXIMISE) {
    const v = adapter.limits[key];
    if (typeof v === 'number') requiredLimits[key as string] = v;
  }

  let device: GPUDevice;
  try {
    device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
  } catch (err) {
    // Some implementations reject very large limits; retry with defaults — on a FRESH adapter:
    // an adapter may create only one device (WebKit: a second requestDevice on it yields an
    // already-lost device; spec: the adapter is "consumed"), docs/research/ios-webgpu.md §2.
    console.warn('[gpu] requestDevice with max limits failed, retrying with defaults', err);
    const fresh = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!fresh) return null;
    adapter = fresh;
    device = await fresh.requestDevice({ requiredFeatures: OPTIONAL_FEATURES.filter((f) => fresh.features.has(f)) });
  }

  device.lost.then((info) => {
    console.warn('[gpu] device lost:', info.reason, info.message);
    contextPromise = null;
    for (const cb of lostListeners) {
      try {
        cb(info);
      } catch (e) {
        console.error(e);
      }
    }
  });
  // Safari 26.x delivers at most one event per addEventListener registration
  // (docs/research/ios-webgpu.md §4.3), so re-arm the listener inside itself.
  const onUncaptured = (ev: Event) => {
    console.error('[gpu] uncaptured error:', (ev as GPUUncapturedErrorEvent).error?.message);
    device.removeEventListener('uncapturederror', onUncaptured);
    device.addEventListener('uncapturederror', onUncaptured);
  };
  device.addEventListener('uncapturederror', onUncaptured);

  const rawInfo = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
  const info = {
    vendor: rawInfo?.vendor ?? '',
    architecture: rawInfo?.architecture ?? '',
    description: rawInfo?.description ?? '',
  };
  const isSoftware = /swiftshader|llvmpipe|lavapipe/i.test(`${info.vendor} ${info.architecture} ${info.description}`);

  return {
    adapter,
    device,
    features: new Set(device.features as unknown as Iterable<string>),
    limits: device.limits,
    info,
    isSoftware,
  };
}

/** Round `n` up to a multiple of `m`. */
export function alignTo(n: number, m: number): number {
  return Math.ceil(n / m) * m;
}

/** Create a GPU buffer, optionally initialised from typed-array data. */
export function createBuffer(
  device: GPUDevice,
  sizeOrData: number | ArrayBufferView,
  usage: GPUBufferUsageFlags,
  label?: string,
): GPUBuffer {
  if (typeof sizeOrData === 'number') {
    return device.createBuffer({ size: alignTo(Math.max(4, sizeOrData), 4), usage, label });
  }
  const data = sizeOrData;
  const size = alignTo(Math.max(4, data.byteLength), 4);
  const buffer = device.createBuffer({ size, usage, label, mappedAtCreation: true });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  buffer.unmap();
  return buffer;
}

/*
 * Submission, readback and error-scope helpers live with their users (ARCHITECTURE §2.1):
 * src/features/gpu-common.ts (`BudgetEncoder`: one command buffer in flight, `Staging`,
 * `compilePipeline`, `errorScope`) and the band loop of src/dense/patchmatch-gpu.ts.
 */
