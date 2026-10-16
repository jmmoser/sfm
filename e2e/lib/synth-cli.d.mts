export interface CliOptions {
  scenes: ('object' | 'yard')[];
  set: string[];
  frames?: number; fps?: number; duration?: number; width?: number; height?: number;
  out?: string; name?: string; seed?: number; blur?: number; noNoise?: boolean;
  depthEvery?: number; gtPoints?: number; gtFocusPoints?: number; timeOffsetMs?: number;
  textureScale?: number; visibilityEvery?: number; supersample?: number; help?: boolean;
}
export const SCENE_DEFAULTS: Record<'object' | 'yard', { duration: number; fps: number }>;
export const USAGE: string;
export class CliError extends Error {}
export function parseArgs(argv: string[]): CliOptions;
export function sceneConfig(scene: 'object' | 'yard', o: CliOptions, warn?: (msg: string) => void): Record<string, unknown> & { scene: string; fps: number; duration: number; frames: number; name?: string };
