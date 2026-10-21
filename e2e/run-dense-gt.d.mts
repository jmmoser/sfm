import type { Page } from 'playwright';
import type { LoadedDataset } from './lib/metrics-core.mjs';

export const CHROMIUM_ARGS: string[];
export interface DenseGTArgs {
  scenes: string[];
  /** DenseGTConfig overrides without `dataset` (src/harness/dense-gt/main.ts). */
  cfg: Record<string, unknown> & { frames?: number[] };
  render: boolean;
  metrics: boolean;
  verbose: boolean;
  out: string | null;
  tag: string | null;
  loadMaps?: string;
  help?: boolean;
}
export function parseArgs(argv: string[]): DenseGTArgs;
export function openDensePage(opts: { verbose?: boolean; outDir: string }): Promise<{ page: Page; close: () => Promise<void> }>;
export function centreOf(pose: { R: ArrayLike<number>; t: ArrayLike<number> }): [number, number, number];
export function parsePlyColors(u8: Uint8Array): Uint8Array | null;
export function breakdown(dataset: LoadedDataset, points: ArrayLike<number>, centres: number[][], tau: number, compVis: Uint8Array | null, colors?: Uint8Array | null): Record<string, unknown>;
export function loadSubsetVisibility(file: string, produced: boolean, nSamples: number): Promise<Uint8Array | null>;
