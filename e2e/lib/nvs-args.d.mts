export const NVS_RENDERERS: string[];
export const NVS_USAGE: string;
export class NvsArgError extends Error {}
export interface NvsArgs {
  dataset: string | null;
  runDir: string | null;
  renderers: string[] | null;
  views: number;
  frames: number[] | null;
  maxSize: number | 'dense' | 'full';
  pose: 'relative' | 'sim3';
  focal: 'relative' | 'keyframe' | 'shared';
  rs: 'auto' | 'run' | 'gt' | 'none';
  keepBlurred: boolean;
  background: number[];
  ssim: boolean;
  saveImages: boolean;
  out: string | null;
  verbose: boolean;
  headed: boolean;
  help: boolean;
}
export function parseNvsArgs(argv: string[]): NvsArgs;
