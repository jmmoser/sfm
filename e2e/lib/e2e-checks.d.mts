import type { Metrics, ObjectRoiJSON } from './metrics-core.mjs';
import type { SelfMetrics } from './self-metrics.mjs';

export interface MetricsInput {
  intrinsics: { f: number; cx: number; cy: number; k1: number; k2: number; width: number; height: number } | null;
  images: { index: number; pose: { R: number[]; t: number[] } }[];
  metric: { metric: boolean; gravityAligned: boolean; northAligned: boolean; source?: string; relativeUncertainty?: number | null };
  numInputImages: number | null;
  pointsFile?: string;
  /** The mesh's object box (null = none proposed); absent without a mesh. */
  objectRoi?: ObjectRoiJSON | null;
}
export interface Limit { min?: number; max?: number; equals?: boolean }
export interface Thresholds { limits: Record<string, Limit>; claims: Record<string, boolean> }
export interface CheckResult { name: string; label: string; value: number | boolean | null; limit: Limit; status: 'pass' | 'fail' | 'skip'; reason?: string }
export interface CheckContext {
  dense: boolean;
  /** Mesh requested (mesh checks apply). */
  mesh?: boolean;
  /** Mesh requested with the object box on (`MeshOptions.objectRoi` not false: object-box checks apply). */
  objectRoi?: boolean;
  claims: { metric?: boolean; gravityAligned?: boolean; northAligned?: boolean } | null;
  sensors: { imu: boolean; gps: boolean; heading: boolean };
}
export function metricsInput(r: unknown, opts?: { pointsFile?: string | null }): MetricsInput;
export function resolveThresholds(spec: unknown, scene: string, name: string): Thresholds;
export const CHECKS: Record<string, { label: string; when: 'always' | 'dense' | 'mesh' | 'objectRoi' | 'metric' | 'gravity' | 'north'; value: (m: Metrics) => number | null }>;
export function runChecks(metrics: Metrics, thresholds: Thresholds, ctx: CheckContext): CheckResult[];
export function skipChecks(thresholds: Thresholds, reason: string): CheckResult[];
export function formatChecks(checks: CheckResult[]): string;
export function table(rows: (string | number)[][], alignLeft?: number[]): string;
export function summaryRow(name: string, r: unknown, m: Metrics | SelfMetrics | null, checks: CheckResult[]): string[];
export const SUMMARY_HEADER: string[];
export function fmtNum(v: unknown, d?: number): string;
