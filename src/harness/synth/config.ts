/**
 * Dataset generation configuration (defaults per scene) and derived quantities
 * (intrinsics, frame times, exposure gain). Pure — usable from Node tests.
 */
import type { CameraIntrinsics } from '../../core/types';
import { focalFromFov, type ScreenAngle } from './conventions';
import { DEFAULT_GPS_CONFIG, type GpsConfig } from './gps';
import { DEFAULT_HEADING_CONFIG, type HeadingConfig } from './heading';
import { DEFAULT_IMU_CONFIG, type ImuConfig } from './imu';
import { DEFAULT_RENDER_EFFECTS, type RenderEffects } from './render-effects';
import { SynthRng } from './rng';
import {
  createObjectTrajectory,
  createYardTrajectory,
  type ObjectTrajectoryOptions,
  type Trajectory,
  type YardTrajectoryOptions,
} from './trajectory';

export type SynthSceneName = 'object' | 'yard';

export interface SynthConfig {
  scene: SynthSceneName;
  /** Dataset name (output directory). Default = scene. */
  name: string;
  width: number;
  height: number;
  fps: number;
  /** Trajectory duration covered by the frames (s). */
  duration: number;
  /** Number of frames (default round(fps·duration)); frames are at τ_i = i / fps. */
  frames: number;
  /** performance.now()-like timestamp (ms) of trajectory time τ = 0 (IMU clock). */
  startTimeMs: number;
  /** Camera–IMU time offset: reported frame t = true t + timeOffsetMs. */
  timeOffsetMs: number;
  seed: number;
  /** Horizontal field of view of the LONG image side (deg); sets f. */
  hfovLongDeg: number;
  /** Principal point offset from the image centre ((W−1)/2, (H−1)/2), px. */
  principalPointOffset: [number, number];
  k1: number;
  k2: number;
  effects: RenderEffects;
  /** Exposure gain jitter amplitude (0.05 = ±5 %). */
  exposureJitter: number;
  jpegQuality: number;
  imu: ImuConfig;
  /** null = no GPS. */
  gps: GpsConfig | null;
  heading: HeadingConfig;
  /** Write GT depth for every N-th frame (0 = never). */
  depthEvery: number;
  /** Number of area-uniform GT surface samples over the whole scene. */
  gtPoints: number;
  /** Extra area-uniform GT samples over the object-of-interest meshes only (0 = none). */
  gtFocusPoints: number;
  /** Visibility test every N-th frame (0 = skip; then gt_visibility is not written). */
  visibilityEvery: number;
  visibilityScale: number;
  textureScale: number;
  anisotropy: number;
  trajectory: Partial<ObjectTrajectoryOptions & YardTrajectoryOptions>;
}

export type SynthConfigInput = Partial<Omit<SynthConfig, 'effects' | 'imu' | 'gps' | 'heading'>> & {
  scene: SynthSceneName;
  effects?: Partial<RenderEffects>;
  imu?: Partial<ImuConfig>;
  gps?: Partial<GpsConfig> | null;
  heading?: Partial<HeadingConfig>;
};

export function resolveConfig(input: SynthConfigInput): SynthConfig {
  const isObject = input.scene === 'object';
  const width = input.width ?? (isObject ? 720 : 960);
  const height = input.height ?? (isObject ? 960 : 720);
  const fps = input.fps ?? (isObject ? 8 : 6);
  const duration = input.duration ?? (isObject ? 25 : 40);
  const frames = input.frames ?? Math.max(1, Math.round(fps * duration));
  const effects: RenderEffects = {
    ...DEFAULT_RENDER_EFFECTS,
    exposureMs: isObject ? 12 : 4,
    ...(input.effects ?? {}),
  };
  const gpsIn = input.gps === undefined ? (isObject ? null : {}) : input.gps;
  return {
    scene: input.scene,
    name: input.name ?? input.scene,
    width,
    height,
    fps,
    duration,
    frames,
    startTimeMs: input.startTimeMs ?? 10000,
    timeOffsetMs: input.timeOffsetMs ?? 25,
    seed: input.seed ?? 1,
    hfovLongDeg: input.hfovLongDeg ?? 65,
    principalPointOffset: input.principalPointOffset ?? (isObject ? [1.5, -1.0] : [-1.0, 1.5]),
    k1: input.k1 ?? -0.03,
    k2: input.k2 ?? 0.005,
    effects,
    exposureJitter: input.exposureJitter ?? 0.05,
    jpegQuality: input.jpegQuality ?? 0.92,
    imu: { ...DEFAULT_IMU_CONFIG, ...(input.imu ?? {}) },
    gps: gpsIn ? { ...DEFAULT_GPS_CONFIG, ...gpsIn } : null,
    heading: { ...DEFAULT_HEADING_CONFIG, ...(input.heading ?? {}) },
    depthEvery: input.depthEvery ?? 20,
    gtPoints: input.gtPoints ?? (isObject ? 1_000_000 : 2_000_000),
    gtFocusPoints: input.gtFocusPoints ?? (isObject ? 1_000_000 : 0),
    visibilityEvery: input.visibilityEvery ?? 1,
    visibilityScale: input.visibilityScale ?? 0.25,
    textureScale: input.textureScale ?? 1,
    anisotropy: input.anisotropy ?? 8,
    trajectory: input.trajectory ?? {},
  };
}

export function screenAngleFor(cfg: SynthConfig): ScreenAngle {
  return cfg.scene === 'object' ? 0 : 90;
}

export function intrinsicsFor(cfg: SynthConfig): CameraIntrinsics {
  const f = focalFromFov(Math.max(cfg.width, cfg.height), cfg.hfovLongDeg);
  return {
    width: cfg.width,
    height: cfg.height,
    f,
    cx: (cfg.width - 1) / 2 + cfg.principalPointOffset[0],
    cy: (cfg.height - 1) / 2 + cfg.principalPointOffset[1],
    k1: cfg.k1,
    k2: cfg.k2,
  };
}

export function trajectoryFor(cfg: SynthConfig): Trajectory {
  const sa = screenAngleFor(cfg);
  if (cfg.scene === 'object') return createObjectTrajectory({ duration: cfg.duration, ...cfg.trajectory, screenAngle: sa });
  return createYardTrajectory({ duration: cfg.duration, ...cfg.trajectory, screenAngle: sa });
}

/** Trajectory time (s) of frame i. */
export function frameTime(cfg: SynthConfig, i: number): number {
  return i / cfg.fps;
}

/** Smooth auto-exposure drift within ±exposureJitter plus ±0.5 % per-frame flicker. */
export function exposureGainFn(cfg: SynthConfig): (tau: number, frame: number) => number {
  const rng = new SynthRng(cfg.seed).derive('exposure');
  const p1 = rng.next() * 6.283,
    p2 = rng.next() * 6.283;
  const J = cfg.exposureJitter;
  return (tau, frame) => {
    const fr = new SynthRng((cfg.seed * 7919 + frame * 104729) >>> 0).next() * 2 - 1;
    return 1 + J * (0.6 * Math.sin((2 * Math.PI * tau) / 7 + p1) + 0.3 * Math.sin((2 * Math.PI * tau) / 2.9 + p2) + 0.1 * fr);
  };
}
