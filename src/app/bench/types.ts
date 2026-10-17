/**
 * On-device benchmark & self-test: result / report types and the message protocol between the
 * Diagnostics screen (bench.ts) and the benchmark worker (bench.worker.ts → runner.ts).
 */

/**
 * Input sizes: 'full' = the sizes the app works at (1920×1440 SIFT, 4096×4096 matching,
 * 640×480 PatchMatch with 8 sources, 100-image BA); 'small' = CI-sized (headless SwiftShader is a
 * CPU emulation of a GPU); 'tiny' = Node unit tests (CPU only).
 */
export type BenchScale = 'full' | 'small' | 'tiny';

export type BenchTestId = 'sift' | 'matcher' | 'patchmatch' | 'ba';

export type BenchStatus = 'pending' | 'running' | 'pass' | 'fail' | 'skipped' | 'cancelled' | 'error';

export interface BenchCheck {
  name: string;
  pass: boolean;
  /** Measured value vs threshold, human readable. */
  detail: string;
}

export interface BenchTestResult {
  id: BenchTestId;
  title: string;
  status: BenchStatus;
  /** What was measured: the GPU path (plus a CPU reference), or only the CPU fallback. */
  backend: 'gpu' | 'cpu';
  /** One line for the UI. */
  summary: string;
  /** Timings in ms by stage name (insertion order = display order). */
  timings: Record<string, number>;
  /** Counts, sizes, accuracy / parity values. */
  metrics: Record<string, number>;
  checks: BenchCheck[];
  /** Why the test was skipped / what was not measured (e.g. no WebGPU). */
  note?: string;
  error?: string;
  /** Wall time of the whole test (ms). */
  ms: number;
}

/** Per-unit costs extracted from the measurements (inputs of the scan-time estimate). */
export interface BenchRates {
  /** SIFT extraction, ms per megapixel (GPU incl. upload and readback; CPU reference). */
  siftGpuMsPerMP: number | null;
  siftCpuMsPerMP: number | null;
  /** Brute-force matching, ms per million descriptor pairs (n_a × n_b). */
  matchGpuMsPerMPairs: number | null;
  matchCpuMsPerMPairs: number | null;
  /** PatchMatch, ms per (megapixel × source × iteration). */
  pmGpuMsPerUnit: number | null;
  pmCpuMsPerUnit: number | null;
  /** Bundle adjustment, ms per LM iteration per 1000 observations. */
  baMsPerIterPerKObs: number | null;
}

export interface BenchEstimate {
  quality: 'fast' | 'balanced' | 'max';
  /** Assumed scan size. */
  keyframes: number;
  /** Live analysis of one frame (features + matching to the last keyframe). */
  frameMs: number;
  /** Frames per second the engine keeps up with (the camera pump runs at ≤ 10 Hz). */
  captureFps: number;
  /** Background mapping work per keyframe (pair matching + local BA). */
  keyframeMs: number;
  refineMs: number;
  denseMs: number;
  meshMs: number;
  /** refine + dense + mesh. */
  processingMs: number;
  features: 'gpu' | 'cpu';
  dense: 'gpu' | 'cpu';
}

export interface BenchDevice {
  userAgent: string;
  hardwareConcurrency: number | null;
  gpu: {
    available: boolean;
    adapter?: string;
    isSoftware?: boolean;
    maxBufferSize?: number;
    maxStorageBufferBindingSize?: number;
    /** Why there is no GPU result. */
    reason?: string;
  };
}

export interface BenchReport {
  version: 1;
  /** ISO time the run started. */
  startedAt: string;
  durationMs: number;
  scale: BenchScale;
  cancelled: boolean;
  device: BenchDevice;
  tests: BenchTestResult[];
  rates: BenchRates;
  estimates: BenchEstimate[];
  /** Assumptions of the estimate model. */
  assumptions: string[];
  /** The run ended abnormally (worker crashed, terminated after a cancel, page hidden). */
  error?: string;
}

// ---- worker protocol --------------------------------------------------------------------------

export type BenchToWorker = { type: 'run'; scale: BenchScale; gpu: boolean; noGpuReason?: string } | { type: 'cancel' };

export type BenchFromWorker =
  | { type: 'test'; result: BenchTestResult }
  | { type: 'progress'; id: BenchTestId; message: string }
  | { type: 'done'; report: BenchReport }
  | { type: 'fatal'; message: string };

export const BENCH_TESTS: readonly { id: BenchTestId; title: string }[] = [
  { id: 'sift', title: 'SIFT features' },
  { id: 'matcher', title: 'Feature matching' },
  { id: 'patchmatch', title: 'Dense stereo (PatchMatch)' },
  { id: 'ba', title: 'Bundle adjustment' },
];
