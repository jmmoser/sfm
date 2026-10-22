export const NVS_RENDERERS: string[];
export const NVS_USAGE: string;
export class NvsArgError extends Error {}
export interface NvsArgs {
  dataset: string | null;
  runDir: string | null;
  testDataset: string | null;
  renderers: string[] | null;
  views: number;
  frames: number[] | null;
  maxSize: number | 'dense' | 'full';
  pose: 'relative' | 'sim3';
  focal: 'relative' | 'keyframe' | 'shared';
  rs: 'auto' | 'run' | 'gt' | 'none';
  rsTranslation: boolean;
  keepBlurred: boolean;
  background: number[];
  ssim: boolean;
  common: boolean;
  /** The splat model file in the run directory (splats.ply; *.spz is decoded first). */
  splatFile: string;
  saveImages: boolean;
  out: string | null;
  verbose: boolean;
  headed: boolean;
  help: boolean;
}
export function parseNvsArgs(argv: string[]): NvsArgs;
