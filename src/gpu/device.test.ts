import { afterEach, describe, expect, it, vi } from 'vitest';

// Node unit tests of the device-request policy with a fake `navigator.gpu` (the real WebGPU
// paths are covered by the *.gpu.test.ts suites in headless Chromium).

function fakeDevice(): GPUDevice {
  return {
    lost: new Promise(() => undefined),
    features: new Set(['shader-f16']),
    limits: {},
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
  } as unknown as GPUDevice;
}

function fakeAdapter(features: string[], limits: Record<string, unknown>, requestDevice: (d: GPUDeviceDescriptor) => Promise<GPUDevice>) {
  return { features: new Set(features), limits, info: { vendor: 'apple', architecture: '', description: '' }, requestDevice: vi.fn(requestDevice) };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetModules();
});

describe('getGPU device request', () => {
  it('requests only features the adapter has and only numeric limits (Safari ≤ 27 rejects undefined)', async () => {
    const a = fakeAdapter(['shader-f16', 'float32-filterable'], { maxBufferSize: 1 << 30, maxBindGroups: 11, maxComputeWorkgroupStorageSize: undefined }, async () => fakeDevice());
    vi.stubGlobal('navigator', { gpu: { requestAdapter: vi.fn(async () => a) } });
    const { getGPU } = await import('./device');
    const ctx = await getGPU();
    expect(ctx).not.toBeNull();
    const desc = a.requestDevice.mock.calls[0][0];
    expect(desc.requiredFeatures).toEqual(['shader-f16', 'float32-filterable']);
    expect(desc.requiredLimits).toEqual({ maxBufferSize: 1 << 30, maxBindGroups: 11 });
  });

  it('retries with default limits on a FRESH adapter (an adapter may create one device only)', async () => {
    const a1 = fakeAdapter(['shader-f16', 'subgroups'], { maxBufferSize: 1 << 30 }, async () => {
      throw new Error('OperationError: limit exceeds the adapter');
    });
    const a2 = fakeAdapter(['shader-f16'], { maxBufferSize: 1 << 28 }, async () => fakeDevice());
    const queue = [a1, a2];
    const requestAdapter = vi.fn(async () => queue.shift() ?? null);
    vi.stubGlobal('navigator', { gpu: { requestAdapter } });
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const { getGPU } = await import('./device');
    const ctx = await getGPU();
    warn.mockRestore();
    expect(ctx).not.toBeNull();
    expect(requestAdapter).toHaveBeenCalledTimes(2);
    expect(a1.requestDevice).toHaveBeenCalledTimes(1);
    expect(a2.requestDevice).toHaveBeenCalledTimes(1);
    expect(a2.requestDevice.mock.calls[0][0]).toEqual({ requiredFeatures: ['shader-f16'] });
    expect(ctx!.adapter).toBe(a2 as unknown as GPUAdapter);
  });

  it('resolves null when no second adapter is available for the retry', async () => {
    const a1 = fakeAdapter([], {}, async () => {
      throw new Error('OperationError');
    });
    const queue = [a1];
    vi.stubGlobal('navigator', { gpu: { requestAdapter: vi.fn(async () => queue.shift() ?? null) } });
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const { getGPU } = await import('./device');
    expect(await getGPU()).toBeNull();
    warn.mockRestore();
  });
});
