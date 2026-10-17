/**
 * Input sizes of the benchmark per scale (shared by the runner and the Diagnostics screen, which
 * describes the tests before a run — this module stays free of the heavy compute imports).
 */
import type { BenchScale, BenchTestId } from './types';

export interface ScaleConfig {
  /** GPU SIFT timing image and timed runs (after one warm-up). */
  sift: [number, number];
  siftRuns: number;
  /** GPU vs CPU parity / CPU speed image. */
  parity: [number, number];
  /** Descriptors per set (matching n × n). */
  match: number;
  pm: { w: number; h: number; sources: number; iterations: number };
  /** GPU vs CPU agreement / CPU speed scene. */
  pmParity: { w: number; h: number; sources: number; iterations: number };
  ba: { poses: number; points: number; iterations: number };
}

export const SCALES: Readonly<Record<BenchScale, ScaleConfig>> = {
  full: {
    sift: [1920, 1440],
    siftRuns: 3,
    parity: [640, 480],
    match: 4096,
    pm: { w: 640, h: 480, sources: 8, iterations: 4 },
    pmParity: { w: 160, h: 120, sources: 4, iterations: 2 },
    ba: { poses: 100, points: 20000, iterations: 8 },
  },
  small: {
    sift: [480, 360],
    siftRuns: 1,
    parity: [320, 240],
    match: 1024,
    pm: { w: 160, h: 120, sources: 4, iterations: 2 },
    pmParity: { w: 160, h: 120, sources: 4, iterations: 2 },
    ba: { poses: 30, points: 3000, iterations: 5 },
  },
  tiny: {
    sift: [240, 180],
    siftRuns: 1,
    parity: [160, 120],
    match: 256,
    pm: { w: 96, h: 72, sources: 3, iterations: 2 },
    pmParity: { w: 96, h: 72, sources: 3, iterations: 2 },
    ba: { poses: 12, points: 500, iterations: 3 },
  },
};

/** What a test measures at `scale` (shown before / while it runs). */
export function describeTest(id: BenchTestId, scale: BenchScale): string {
  const c = SCALES[scale];
  switch (id) {
    case 'sift':
      return `WebGPU SIFT on a ${c.sift[0]}×${c.sift[1]} image, timing per stage; GPU vs CPU parity at ${c.parity[0]}×${c.parity[1]}`;
    case 'matcher':
      return `${c.match} × ${c.match} descriptors on the GPU vs the CPU matcher (identical pairs) and the true correspondences`;
    case 'patchmatch':
      return `WebGPU PatchMatch ${c.pm.w}×${c.pm.h}, ${c.pm.sources} sources, ${c.pm.iterations} iterations vs ground-truth depth; CPU agreement`;
    case 'ba':
      return `Synthetic ${c.ba.poses}-image problem, ${c.ba.points} points: time per Levenberg–Marquardt iteration`;
  }
}
