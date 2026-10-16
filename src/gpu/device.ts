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
  'maxStorageBuffersPerShaderStage',
  'maxStorageTexturesPerShaderStage',
  'maxSampledTexturesPerShaderStage',
  'maxTextureDimension2D',
  'maxTextureArrayLayers',
  'maxComputeWorkgroupsPerDimension',
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
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return null;

  const requiredFeatures = OPTIONAL_FEATURES.filter((f) => adapter.features.has(f));
  const requiredLimits: Record<string, number> = {};
  for (const key of LIMITS_TO_MAXIMISE) {
    const v = adapter.limits[key];
    if (typeof v === 'number') requiredLimits[key as string] = v;
  }

  let device: GPUDevice;
  try {
    device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
  } catch (err) {
    // Some implementations reject very large limits; retry with defaults.
    console.warn('[gpu] requestDevice with max limits failed, retrying with defaults', err);
    device = await adapter.requestDevice({ requiredFeatures });
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
  device.addEventListener('uncapturederror', (ev) => {
    console.error('[gpu] uncaptured error:', (ev as GPUUncapturedErrorEvent).error.message);
  });

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

/**
 * Copy `size` bytes from a GPU buffer (needs COPY_SRC usage) to the CPU.
 * Returns a fresh ArrayBuffer that the caller owns.
 */
export async function readBuffer(device: GPUDevice, src: GPUBuffer, size: number, srcOffset = 0): Promise<ArrayBuffer> {
  const alignedSize = alignTo(size, 4);
  const staging = device.createBuffer({
    size: alignedSize,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    label: 'readback-staging',
  });
  const enc = device.createCommandEncoder({ label: 'readback' });
  enc.copyBufferToBuffer(src, srcOffset, staging, 0, alignedSize);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = staging.getMappedRange(0, alignedSize).slice(0, size);
  staging.unmap();
  staging.destroy();
  return out;
}

/**
 * Small cache for compiled compute pipelines keyed by a string, so modules can
 * lazily create pipelines without recompiling identical WGSL.
 */
export class PipelineCache {
  private readonly cache = new Map<string, GPUComputePipeline>();
  constructor(private readonly device: GPUDevice) {}

  get(key: string, code: string, entryPoint = 'main', constants?: Record<string, number>): GPUComputePipeline {
    const fullKey = constants ? `${key}|${JSON.stringify(constants)}` : key;
    let p = this.cache.get(fullKey);
    if (!p) {
      const module = this.device.createShaderModule({ code, label: key });
      p = this.device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint, constants },
        label: key,
      });
      this.cache.set(fullKey, p);
    }
    return p;
  }

  clear(): void {
    this.cache.clear();
  }
}

/**
 * Await GPU queue completion. Useful to bound the amount of queued work on iOS,
 * where very long command buffers can trip the GPU watchdog: split heavy work into
 * several submits and await between them.
 */
export function gpuIdle(device: GPUDevice): Promise<void> {
  return device.queue.onSubmittedWorkDone();
}

/** Run `fn` inside an error scope and throw if a validation/OOM error occurred. */
export async function withErrorScope<T>(device: GPUDevice, label: string, fn: () => T): Promise<T> {
  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('validation');
  const result = fn();
  const validation = await device.popErrorScope();
  const oom = await device.popErrorScope();
  if (validation) throw new Error(`[gpu:${label}] validation error: ${validation.message}`);
  if (oom) throw new Error(`[gpu:${label}] out of memory: ${oom.message}`);
  return result;
}
