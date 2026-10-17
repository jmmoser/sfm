// Node unit tests of compilePipeline's Safari safeguards (fake device): a stalled
// createComputePipelineAsync falls back to the synchronous compile; device loss wins the race.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { GPUFeatureError, compilePipeline, pipelineCompileLimits } from './gpu-common';

const defaultTimeout = pipelineCompileLimits.timeoutMs;

afterEach(() => {
  pipelineCompileLimits.timeoutMs = defaultTimeout;
});

function fakeDevice(opts: { asyncNeverSettles?: boolean; syncError?: string }) {
  const scopes: (GPUError | null)[][] = [];
  const syncPipe = { label: 'sync' } as unknown as GPUComputePipeline;
  const asyncPipe = { label: 'async' } as unknown as GPUComputePipeline;
  const device = {
    pushErrorScope: () => scopes.push([]),
    popErrorScope: async () => scopes.pop()?.[0] ?? null,
    createShaderModule: () => ({ getCompilationInfo: async () => ({ messages: [] }) }),
    createComputePipelineAsync: vi.fn(() => (opts.asyncNeverSettles ? new Promise<GPUComputePipeline>(() => undefined) : Promise.resolve(asyncPipe))),
    createComputePipeline: vi.fn(() => {
      if (opts.syncError) scopes[scopes.length - 1]?.push({ message: opts.syncError } as GPUError);
      return syncPipe;
    }),
  };
  return { device: device as unknown as GPUDevice, raw: device, syncPipe, asyncPipe };
}

describe('compilePipeline (Safari async-compile safeguards)', () => {
  it('uses the async pipeline when it settles', async () => {
    const f = fakeDevice({});
    const p = await compilePipeline(f.device, 'k', 'code', new Promise(() => undefined));
    expect(p).toBe(f.asyncPipe);
    expect(f.raw.createComputePipeline).not.toHaveBeenCalled();
  });

  it('falls back to the synchronous compile when the async one never settles', async () => {
    pipelineCompileLimits.timeoutMs = 20;
    const f = fakeDevice({ asyncNeverSettles: true });
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const p = await compilePipeline(f.device, 'k', 'code', new Promise(() => undefined));
    warn.mockRestore();
    expect(p).toBe(f.syncPipe);
    expect(f.raw.createComputePipeline).toHaveBeenCalledTimes(1);
  });

  it('reports a validation error of the synchronous fallback as GPUFeatureError', async () => {
    pipelineCompileLimits.timeoutMs = 20;
    const f = fakeDevice({ asyncNeverSettles: true, syncError: 'bad shader' });
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    await expect(compilePipeline(f.device, 'k', 'code', new Promise(() => undefined))).rejects.toBeInstanceOf(GPUFeatureError);
    warn.mockRestore();
  });

  it('rejects with GPUFeatureError when the device is lost first (no sync compile)', async () => {
    pipelineCompileLimits.timeoutMs = 10_000;
    const f = fakeDevice({ asyncNeverSettles: true });
    await expect(compilePipeline(f.device, 'k', 'code', Promise.resolve())).rejects.toThrow(/device lost/);
    expect(f.raw.createComputePipeline).not.toHaveBeenCalled();
  });
});
