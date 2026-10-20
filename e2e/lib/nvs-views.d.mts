import type { Sim3 } from './linalg.mjs';
export interface PoseJSON { R: number[]; t: number[] }
export interface FrameJSON { index: number; file: string; t?: number; tTrue?: number; pose: PoseJSON; f?: number; gain?: number; wb?: number[]; rsPx?: number; blurPx?: number; omega?: number }
export interface RunImage { index: number; pose: PoseJSON; f?: number }
export interface RollingShutter { axis: 'rows' | 'cols'; readoutMs: number }
export interface TestView {
  index: number;
  file: string;
  tTrue?: number;
  nearestKeyframe: number;
  gap: number;
  pose: PoseJSON;
  camera: { f: number; cx: number; cy: number; k1: number; k2: number; width: number; height: number };
  omega: number[] | null;
  gt: { f: number | null; gain: number | null; wb: number[] | null; rsPx: number | null; blurPx: number | null; omega: number | null };
}
export interface TestViewOptions {
  views?: number;
  pose?: 'relative' | 'sim3';
  focal?: 'relative' | 'keyframe' | 'shared';
  keepBlurred?: boolean;
  frames?: number[];
}
export function cameraCenter(pose: PoseJSON): number[];
export function registeredImages(result: unknown): RunImage[];
export function runToGtSim3(images: RunImage[], framesByIndex: Map<number, FrameJSON>): Sim3;
export function heldOutFrames<T extends { index: number }>(frames: T[], result: unknown, opts?: { keepBlurred?: boolean }): T[];
export function selectEvenly<T>(list: T[], n: number): T[];
export function nearestImage<T extends { index: number }>(sorted: T[], index: number): T | null;
export function mapPoseRelative(gtFrame: PoseJSON, gtKey: PoseJSON, runKey: PoseJSON, s: number): PoseJSON;
export function mapPoseSim3(gtFrame: PoseJSON, sim: Sim3): PoseJSON;
export function gyroAt(imu: { t: number; gyro: number[] }[], t: number): number[] | null;
export function cameraOmega(meta: { R_cam_device?: number[] }, imu: { t: number; gyro: number[] }[], frame: { t?: number; tTrue?: number }): number[] | null;
export function runRollingShutter(result: unknown): RollingShutter | null;
export function gtRollingShutter(meta: unknown): RollingShutter | null;
export function denseImageSize(result: unknown): number | null;
export function testViews(
  dataset: { meta: any; frames: FrameJSON[]; imu?: { t: number; gyro: number[] }[] | null },
  result: any,
  opts?: TestViewOptions,
): { sim3: Sim3; candidates: number; views: TestView[] };
