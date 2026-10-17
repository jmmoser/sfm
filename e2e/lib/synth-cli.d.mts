export type SynthTarget = 'object' | 'yard' | 'object-real' | 'yard-real';
export interface CliOptions {
  scenes: SynthTarget[];
  set: string[];
  frames?: number; fps?: number; duration?: number; width?: number; height?: number;
  out?: string; name?: string; seed?: number; blur?: number; noNoise?: boolean;
  depthEvery?: number; gtPoints?: number; gtFocusPoints?: number; timeOffsetMs?: number;
  textureScale?: number; visibilityEvery?: number; supersample?: number; help?: boolean;
  rsReadoutMs?: number; rsBands?: number; focusBreathing?: number; whiteBalanceDrift?: number;
  exposureJitter?: number; jpegQuality?: number; adaptiveBlurPx?: number;
}
export const PRESET_SCENES: Record<'object-real' | 'yard-real', 'object' | 'yard'>;
export function baseScene(target: string): string;
export const SCENE_DEFAULTS: Record<'object' | 'yard', { duration: number; fps: number }>;
export const USAGE: string;
export class CliError extends Error {}
export function parseArgs(argv: string[]): CliOptions;
export function sceneConfig(target: SynthTarget, o: CliOptions, warn?: (msg: string) => void): Record<string, unknown> & { scene: string; fps: number; duration: number; frames: number; name?: string; preset?: string };
