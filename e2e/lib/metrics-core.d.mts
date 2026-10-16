import type { Sim3 } from './linalg.mjs';

export interface DatasetPose { R: number[]; t: number[] }
export interface DatasetFrame { index: number; t?: number; tTrue?: number; pose: DatasetPose; [k: string]: unknown }
export interface GtSet { points: Float32Array; normals?: Int8Array | null; visibility?: Uint8Array | null }
export interface DatasetMeta {
  name: string;
  scene: string;
  width: number;
  height: number;
  intrinsics: { f: number; cx: number; cy: number; k1: number; k2: number };
  sceneExtent: number;
  roi?: { min: number[]; max: number[] };
  denseTau?: number;
  gt?: { spacing?: number; focus?: { spacing?: number } | null } | null;
  [k: string]: unknown;
}
export interface LoadedDataset { meta: DatasetMeta; frames: DatasetFrame[]; gt?: GtSet | null; gtFocus?: GtSet | null }
export interface ResultJSON {
  intrinsics?: { f: number; cx?: number; cy?: number; k1?: number; k2?: number };
  images: { index: number; pose: DatasetPose }[];
  metric?: { metric: boolean; gravityAligned: boolean; northAligned: boolean };
  pointsFile?: string;
  numInputImages?: number;
}
export interface MetricsOptions {
  points?: Float32Array;
  tau?: number;
  roi?: boolean;
  minVisibility?: number;
  denseAlign?: 'sim3' | 'sim3+icp' | 'se3' | 'none';
  maxPoints?: number;
  maxDistFactor?: number;
}
export interface Metrics {
  dataset: string;
  scene: string;
  numFrames: number;
  registration: { numFrames: number; numRegistered: number; rate: number; numInput: number | null; rateOfInput: number | null; unknownIndices: number };
  trajectory: {
    sim3: Sim3; ateRmse: number; ateMedian: number; ateMax: number; sceneExtent: number; ateRelative: number;
    rotErrDegMean: number; rotErrDegMedian: number; rotErrDegMax: number; rpeRotDegMean: number | null; rpeRotDegMax: number | null;
  } | null;
  metric: { scale: number; scaleError: number; ateSe3Rmse: number | null; ateSe3Relative: number | null } | null;
  frame: { gravityAligned: boolean; northAligned: boolean; gravityErrorDeg: number; gravityErrorDegSim3: number | null; headingErrorDeg: number; headingErrorDegSim3: number | null } | null;
  intrinsics: { fGt: number; fRes: number; focalErrorPct: number; ppErrorPx: number | null; k1Error: number | null; k2Error: number | null; cornerDistortionErrorPx: number | null } | null;
  dense: {
    tau: number; align: string; icp: { iterations: number; rotationDeg: number; translation: number } | null; roi: boolean; numPointsRaw: number; numPointsEvaluated: number;
    accuracyMedian: number; accuracyP90: number; accuracyMeanCapped: number; precision: number;
    completeness: number; completenessSet: 'focus' | 'base'; numGtEvaluated: number; fscore: number;
  } | null;
}
export function computeMetrics(dataset: LoadedDataset, result: ResultJSON, opts?: MetricsOptions): Metrics;
export function formatTable(m: Metrics): string;
