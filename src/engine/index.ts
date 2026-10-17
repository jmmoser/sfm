/**
 * Engine module (docs/ENGINE.md): PhotogrammetryEngine and its building blocks. The public API
 * types (EngineOptions, FrameInput, LiveUpdate, SparseResult, …) are defined by the worker
 * protocol (src/workers/protocol.ts) and re-exported here.
 */
export { PhotogrammetryEngine, pipelineKey } from './engine';
export type { EngineConfig, EngineEvents, EngineStage, KeyframeRecord, SparseCameraEx, SparseModel, SparseResultEx } from './types';
export { QUALITY_PRESETS, ENGINE_DEFAULTS, resolveDensePreset, externalDensePreset, modeConfig } from './presets';
export type { QualityPresetConfig, DensePresetConfig, FeaturePresetConfig, ModeConfig, ResolvedMode } from './presets';
export { MEM, captureMemory, planDense, deviceMemoryBudgetMB, reconstructionBytes, featureSetBytes, formatBreakdown } from './memory';
export type { MemoryBreakdown, DensePlan, DensePlanInput, DenseStagePeak } from './memory';
export { MemoryEngineStore, SessionEngineStore, packSparseModel, unpackSparseModel, packImu, unpackImu } from './persistence';
export type { EngineStore, EngineStateRecord, StoredImage, StoredPair } from './persistence';
export { OrientationTracker, prepareFrame, overlayPoints, gravityDown, sampleFeatureColors, resizeRGBA, encodeJpegRGBA } from './frame-ingest';
export type { OrientationVerdict, PreparedFrame, FrameSource } from './frame-ingest';
export { LiveSession, CaptureState, ImuLog, handEyeCheck, classifyScene } from './live';
export type { HandEyeResult, LiveDeps } from './live';
export { runRefine, modelFromReconstruction, modelPointCloud } from './refine';
export { runDensify, runMesh, PatchMatchHolder, medianSparseDepth, spreadSubset } from './dense';
export { buildExport, colmapModel, encodeBMP } from './export';
export { EngineDepthStore } from './depth-store';
export { LruCache, decodeStoredImage } from './image-cache';
export type {
  EngineOptions,
  FrameInput,
  LiveUpdate,
  SparseResult,
  SparseCamera,
  DenseOptions,
  MeshOptions,
  ExportFormat,
  SessionSummary,
  InitResult,
} from '../workers/protocol';
