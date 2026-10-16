/**
 * Feature extraction and matching. CPU reference implementations live here; the
 * WebGPU implementations (sift-gpu.ts, match-gpu.ts) and the CPU/GPU-selecting
 * pipeline (extractor.ts) are added alongside.
 * Test helpers (testdata.ts) are intentionally not re-exported.
 */
export * from './image';
export * from './sift-common';
export * from './sift-cpu';
export * from './distribute';
export * from './match';
export { simdAvailable } from './simd';
export { GPUFeatureError, GPUWorkDroppedError } from './gpu-common';
export { SiftGPU } from './sift-gpu';
export type { SiftGPUConfig, SiftGPUTimings, SiftBitmapSource, SiftSelfTestResult } from './sift-gpu';
export { MatcherGPU } from './match-gpu';
export type { MatcherGPUConfig, MatcherGPUTimings, MatcherSelfTestResult } from './match-gpu';
export { createFeaturePipeline, bitmapToGrayCPU } from './extractor';
export type { FeaturePipeline, FeaturePipelineOptions } from './extractor';
