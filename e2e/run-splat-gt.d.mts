export interface SplatGTArgs {
  scenes: string[];
  /** SplatGTConfig overrides without `dataset` (src/harness/splat-gt/main.ts). */
  cfg: Record<string, unknown> & { trainFrames?: number[]; evalAt?: number[] };
  verbose: boolean;
  out: string | null;
  tag: string | null;
  help?: boolean;
}
export function parseArgs(argv: string[]): SplatGTArgs;
