export const USAGE: string;
export class ArgError extends Error {}
export interface E2EArgs {
  datasets: string[];
  /** Harness config without `dataset` / `output` (src/harness/engine/types.ts HarnessConfig). */
  config: Record<string, unknown> & { engine: 'worker' | 'inline' | 'mock'; quality: 'fast' | 'balanced' | 'max' };
  dense: boolean;
  mesh: boolean;
  denseOpts: { maxImageSize?: number; maxViews?: number };
  meshOpts: { voxelSize?: number; targetTriangles?: number };
  exports: boolean | string[];
  timeoutMin: number;
  out: string | null;
  tag: string | null;
  thresholds: string | null;
  noThresholds: boolean;
  enforce: boolean;
  quiet: boolean;
  verbose: boolean;
  headed: boolean;
  help: boolean;
  sets: [string, unknown][];
}
export function parseArgs(argv: string[]): E2EArgs;
