/**
 * Dense reconstruction (ARCHITECTURE §4.6): PatchMatch MVS (WebGPU + CPU),
 * synthetic test-data generator. Undistortion, view selection, fusion and TSDF
 * files are added alongside and re-exported here.
 */
export * from './types';
export * from './patchmatch-common';
export { patchMatchCPU, lastPatchMatchCPUStats, type PatchMatchCPUStats } from './patchmatch-cpu';
export { PatchMatchGPU, type PatchMatchGPUTimings } from './patchmatch-gpu';
export { patchMatchWGSL, type PatchMatchShaderConfig } from './patchmatch-wgsl';
export * from './synthetic';
