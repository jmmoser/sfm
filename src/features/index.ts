/**
 * Feature extraction and matching: the CPU reference implementations. The WebGPU modules
 * (sift-gpu.ts, match-gpu.ts) and the CPU/GPU-selecting pipeline (extractor.ts) are NOT
 * re-exported (only their types, which are erased): sift-gpu / match-gpu read `GPUBufferUsage`
 * at module load, which throws where WebGPU does not exist — import them directly (the engine
 * loads extractor.ts lazily).
 * Test helpers (testdata.ts) are intentionally not re-exported.
 */
export * from './image';
export * from './sift-common';
export * from './sift-cpu';
export * from './distribute';
export * from './match';
export { simdAvailable } from './simd';
export { FeatureInputError, GPUFeatureError, GPUWorkDroppedError } from './gpu-common';
export type { SiftGPUConfig, SiftGPUTimings, SiftBitmapSource, SiftSelfTestResult } from './sift-gpu';
export type { MatcherGPUConfig, MatcherGPUTimings, MatcherSelfTestResult } from './match-gpu';
export type { FeaturePipeline, FeaturePipelineOptions } from './extractor';
