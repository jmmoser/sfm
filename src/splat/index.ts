/**
 * 3D Gaussian Splatting: WebGPU trainer and renderer seeded from the metric SfM + MVS
 * reconstruction, a CPU reference, INRIA-layout PLY import / export, initialisation from point
 * clouds and the MCMC densification planner. Nothing here touches WebGPU globals at module load,
 * so the module can be imported where WebGPU is missing. The engine stage is
 * src/engine/splat.ts; the viewer draws models with `SplatRenderer` + `SplatPresenter`
 * (present.ts); the novel-view harness with nvs.ts.
 *
 * Written from the papers (3DGS, EWA splatting, Mip-Splatting, 3DGS-MCMC; Taming-3DGS / Brush /
 * gsplat design notes) — no INRIA-derived code (non-commercial licence).
 *
 * WebGPU limits every kernel keeps (iOS Safari / Apple GPUs): no subgroups, no f32 atomics (float
 * adds are compare-exchange on u32 bits), no decoupled look-back (no forward-progress guarantee),
 * ≤ 256 threads and ≤ 16 KB workgroup memory (raster 12 KB / 14 KB with geometry, raster backward
 * 7.5 / ≈ 10 KB, SSIM 13.7 KB, colour reduction 16 KB), ≤ 8 storage buffers per kernel, no vec3
 * in buffers, every buffer ≤ maxStorageBufferBindingSize (the intersection reservation only grows
 * while it fits; one training image per buffer), all pipelines compiled up front (`create`), one
 * heavy command buffer in flight.
 *
 * Choices that differ from the reference 3DGS: the rasteriser backward re-walks front to back
 * with the forward's final colour (no division by T); MCMC relocation is planned on the host from
 * one opacity readback per refinement and applied on the GPU; the scale regulariser is divided by
 * the scene extent; Adam updates only Gaussians projected into the frame; SSIM uses zero padding
 * (its own adjoint); the projection Jacobian is evaluated at the frustum-clamped point (gsplat).
 *
 * Tests: cpu-reference.test.ts / depth-target.test.ts check every gradient against finite
 * differences; the *.gpu.test.ts files compare the GPU kernels with the CPU reference (Metal with
 * E2E_GPU=host; SwiftShader works but spends ~5–8 min per file compiling pipelines).
 * Measurements and tuning (results, ablations, defaults): docs/research/splat-tuning.md.
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
