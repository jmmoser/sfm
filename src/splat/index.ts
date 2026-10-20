/**
 * 3D Gaussian Splatting (ARCHITECTURE §6 "src/splat"): WebGPU trainer and renderer seeded from
 * the metric SfM + MVS reconstruction, a CPU reference, INRIA-layout PLY import / export,
 * initialisation from point clouds and the MCMC densification planner. Nothing here touches
 * WebGPU globals at module load, so the module can be imported where WebGPU is missing.
 */
export * from './types';
export { SplatRenderer, splatImageToRGBA8, type SplatImage, type SplatRendererOptions, type RenderCallOptions } from './renderer';
export {
  SplatTrainer,
  DEFAULT_SPLAT_LR,
  type SplatTrainingView,
  type SplatTrainerOptions,
  type SplatLearningRates,
  type SplatMcmcOptions,
  type SplatColourCorrection,
  type SplatStepResult,
  type SplatTrainerStats,
} from './trainer';
export { encodeSplatPLY, decodeSplatPLY, type SplatPLYOptions } from './ply';
export { initFromPointCloud, backgroundShell, concatModels, quatFromZ, type SplatInitOptions, type BackgroundShellOptions, type ColourSource } from './init';
export { planMcmc, relocation, type McmcPlan, type McmcPlanOptions } from './mcmc';
export { packParams, unpackParams, paramOffsets, type ParamOffsets } from './layout';
/** CPU reference renderer / backward / loss (tests, small images). */
export * as splatReference from './cpu-reference';
