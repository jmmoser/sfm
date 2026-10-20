import { describe, expect, it } from 'vitest';

describe('features barrel', () => {
  it('loads without WebGPU and exports only the CPU modules', async () => {
    // sift-gpu / match-gpu (and extractor, which imports them) read GPUBufferUsage at module load.
    expect(typeof (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage).toBe('undefined');
    const m: Record<string, unknown> = await import('./index');
    expect(typeof m.matchCPU).toBe('function');
    expect(typeof m.extractSiftCPU).toBe('function');
    expect(typeof m.selectFeatures).toBe('function');
    expect(typeof m.GPUWorkDroppedError).toBe('function');
    for (const name of ['SiftGPU', 'MatcherGPU', 'createFeaturePipeline', 'bitmapToGrayCPU']) expect(m).not.toHaveProperty(name);
  });
});
