/**
 * Dense reconstruction (ARCHITECTURE §4.6): PatchMatch MVS (WebGPU + CPU), undistortion,
 * view selection, fusion, halo / sky filter, TSDF + marching cubes, simplification, the colour
 * atlas of AR exports, point-cloud filters, tuned presets, and the synthetic test-data generator
 * (`synthetic` namespace).
 */
export * from './types';
export { patchMatchCPU, lastPatchMatchCPUStats, type PatchMatchCPUStats } from './patchmatch-cpu';
export { PatchMatchGPU, type PatchMatchGPUTimings, type PatchMatchSelfTestResult } from './patchmatch-gpu';
export { patchMatchWGSL, type PatchMatchShaderConfig } from './patchmatch-wgsl';
export { resolvePatchMatchOptions, planeHomography, buildSourceCamera, type ResolvedPatchMatchOptions, type SourceCamera } from './patchmatch-common';
/** Low-level PatchMatch helpers (PCG, checkerboard pattern, prior resampling, …). */
export * as patchMatchInternals from './patchmatch-common';
/** CPU ray caster + synthetic scenes / datasets / depth-map evaluation (tests, harness). */
export * as synthetic from './synthetic';

// --- fusion & meshing (undistortion, view selection, fusion, TSDF, marching cubes, simplification, filters)
export { undistortImage, undistortedCamera, undistortedSize, radialFoldRadiusSq, areaResampleRGBA, type UndistortOptions } from './undistort';
export {
  selectSourceViews,
  neighboursFromSelection,
  triangulationAngleWeight,
  scaleWeight,
  DEFAULT_VIEW_SELECTION_OPTIONS,
  type ViewSelectionOptions,
  type ViewSelectionView,
  type SourceViewSelection,
  type SparsePointsInput,
} from './view-selection';
export { fuseDepthMaps, DEFAULT_FUSION_OPTIONS, type FusionOptions, type FusionStats } from './fusion';
export { depthHaloMask, removeDepthHalos, worldUpInCamera, skyBorders, DEFAULT_HALO_FILTER_OPTIONS, type HaloFilterOptions } from './depth-filter';
export { DENSE_PRESETS, type DensePreset, type DenseQuality } from './presets';
export { tsdfMesh, buildTsdfVolume, integrateDepthMap, TsdfVolume, type TsdfOptions, type TsdfStats } from './tsdf';
export {
  marchingCubes,
  removeSmallComponents,
  meshComponents,
  meshEdgeStats,
  compactMesh,
  BLOCK_SIZE,
  BLOCK_VOXELS,
  TSDF_SCALE,
  WEIGHT_UNIT,
  COLOR_SCALE,
  type SparseVoxelVolume,
  type MarchingCubesOptions,
} from './marching-cubes';
export { simplifyMesh, type SimplifyOptions } from './simplify';
export { buildColourAtlas, buildColourAtlasAsync, ClosestTriangleFinder, type ColourAtlas, type ColourAtlasOptions, type ColourAtlasStats } from './colour-atlas';
export {
  filterPointCloud,
  voxelDownsample,
  statisticalOutlierMask,
  selectPoints,
  type PointCloudFilterOptions,
  type StatisticalFilterOptions,
} from './pointcloud';
export { KdTree3 } from './kdtree';
export { DenseCancelledError, DepthMapCache } from './dense-util';
