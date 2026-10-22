export interface GyroBreaks { breaks: number; outliers: number; medianDeg: number }
export interface SelfMetrics {
  dataset: string;
  scene: string;
  groundTruth: false;
  numFrames: number;
  registration: { numFrames: number; delivered: number; keyframes: number; liveRegistered: number | null; numInput: number | null; numRegistered: number; rateOfInput: number | null };
  reprojection: { meanPx: number | null };
  tracks: { mean: number | null; median: number | null; points: number | null };
  gyro: {
    model: string | null;
    breaks: number | null;
    breakOutliers: number | null;
    medianErrDeg: number | null;
    live: GyroBreaks | null;
    offline: GyroBreaks | null;
    deregisteredOutliers: number;
    rejectedPairs: number | null;
    rejectedPoses: number | null;
  };
  explained: { live: number | null; offline: number | null };
  offlineSfM: string | null;
  scale: {
    metric: boolean;
    source: string | null;
    relativeUncertainty: number | null;
    gravityAligned: boolean;
    northAligned: boolean;
    yawSource: string | null;
    gravityStdDeg: number | null;
    timeOffsetMs: number | null;
    warnings: string | null;
  } | null;
  intrinsics: { f: number; k1: number | null; k2: number | null; fPrior: number | null; focalVsPriorPct: number | null } | null;
  live: { timeToInit: unknown; trackingLostFrames: number; frameMsMedian: number | null; frameMsP90: number | null; decisions: Record<string, number> };
  dense: { points: number } | null;
  mesh: { triangles: number } | null;
  device: Record<string, unknown> | null;
  agreement: { compared: number; same: number; rate: number | null; keyframesDevice: number; keyframesReplay: number; confusion: Record<string, number> } | null;
  trajectory: null;
  metric: null;
  frame: null;
}
export function isGtFree(meta: unknown, frames: unknown): boolean;
export function parseGyroBreaks(s: unknown): GyroBreaks | null;
export function computeSelfMetrics(r: unknown, meta?: unknown): SelfMetrics;
export function formatSelfTable(m: SelfMetrics): string;
