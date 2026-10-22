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
  /** Run the Gaussian-splatting stage (implied by any --splat-* option). */
  splat: boolean;
  splatOpts: { iterations?: number; maxGaussians?: number; maxImageSize?: number; maxViews?: number };
  exports: boolean | string[];
  /** Run the novel-view evaluation (e2e/nvs.mjs) on the output. */
  nvs: boolean;
  nvsViews: number;
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
  /** Where the harness runs: here (Playwright), a phone through the HTTP bridge, or the bridge driven by local Chromium. */
  device: 'desktop' | 'phone' | 'local-bridge';
  /** --device phone: HTTPS tunnel URL (no trailing slash). */
  publicUrl: string | null;
  port: number | null;
}
export function parseArgs(argv: string[]): E2EArgs;
