/**
 * Dataset generation configuration (defaults per scene) and derived quantities
 * (intrinsics, frame times, exposure gain). Pure — usable from Node tests.
 */
import type { CameraIntrinsics } from '../../core/types';
import { focalFromFov, type ScreenAngle } from './conventions';
import { DEFAULT_GPS_CONFIG, type GpsConfig } from './gps';
import { DEFAULT_HEADING_CONFIG, type HeadingConfig } from './heading';
import { DEFAULT_IMU_CONFIG, type ImuConfig } from './imu';
import { DEFAULT_FOCUS_BREATHING, type FocusBreathingConfig } from './realism';
import { DEFAULT_RENDER_EFFECTS, type RenderEffects } from './render-effects';
import { SynthRng } from './rng';
import {
  createObjectTrajectory,
  createYardTrajectory,
  type ObjectTrajectoryOptions,
  type ShakeOptions,
  type Trajectory,
  type YardTrajectoryOptions,
} from './trajectory';

export type SynthSceneName = 'object' | 'yard';
export type SynthPresetName = 'object-real' | 'yard-real';

export interface SynthConfig {
  scene: SynthSceneName;
  /** Named preset the config was built from (SYNTH_PRESETS), or null. */
  preset: SynthPresetName | null;
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
  /**
   * Camera–IMU time offset: reported frame t = true t + timeOffsetMs (default +25 ms). IMU, GPS
   * and compass stay on the true clock (`tTrue`).
   */
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
  /**
   * Rolling-shutter readout time (ms) from the first to the last sensor line; 0 = global
   * shutter (default). Direction / sign / timestamp: realism.ts header and e2e/README.md.
   */
  rsReadoutMs: number;
  /** Number of render bands across the readout (default 32); the maximum when rsBandStepPx > 0. */
  rsBands: number;
  /**
   * Adaptive band count: largest image step (px) between neighbouring bands (relaxed to 10 %
   * of the frame's motion blur), see realism.ts rsBandCount. 0 = always rsBands (default).
   */
  rsBandStepPx: number;
  /** Reverse the readout order (default false = native row 0 first). */
  rsReverse: boolean;
  /** Focus breathing (per-frame focal); null = off (default). */
  focusBreathing: FocusBreathingConfig | null;
  /** White-balance drift amplitude (0.03 = ±3 % red/blue swing); 0 = off (default). */
  whiteBalanceDrift: number;
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

export type SynthConfigInput = Partial<Omit<SynthConfig, 'effects' | 'imu' | 'gps' | 'heading' | 'focusBreathing' | 'preset'>> & {
  scene: SynthSceneName;
  /** Apply SYNTH_PRESETS[preset] first; every other field overrides it (objects merge deeply). */
  preset?: SynthPresetName | null;
  effects?: Partial<RenderEffects>;
  imu?: Partial<ImuConfig>;
  gps?: Partial<GpsConfig> | null;
  heading?: Partial<HeadingConfig>;
  /** Partial fields merge into DEFAULT_FOCUS_BREATHING; null = off. */
  focusBreathing?: Partial<FocusBreathingConfig> | null;
};

/**
 * Hand-held shake of the 'object-real' preset: 8–12 Hz tremor + three fast glance pans (own peak
 * rate 1.05–1.22 rad/s; with the orbit motion the body rate peaks at ≈ 1.4–1.5 rad/s).
 */
export const OBJECT_REAL_SHAKE: ShakeOptions = {
  tremorDeg: 0.06,
  tremorM: 0.0001,
  pans: [
    { t: 5.5, yawRad: 0.3, pitchRad: 0.04, sigmaS: 0.15 },
    { t: 12.8, yawRad: -0.26, pitchRad: -0.1, sigmaS: 0.14 },
    { t: 19.6, yawRad: 0.24, pitchRad: 0.1, sigmaS: 0.15 },
  ],
};

/**
 * Hand-held shake of the 'yard-real' preset: 8–12 Hz tremor + four fast glance pans (own peak rate
 * 1.1–1.4 rad/s; with the walk's panning the body rate peaks at ≈ 1.3–1.5 rad/s).
 */
export const YARD_REAL_SHAKE: ShakeOptions = {
  tremorDeg: 0.06,
  tremorM: 0.0001,
  pans: [
    { t: 6, yawRad: 0.55, pitchRad: 0, sigmaS: 0.27 },
    { t: 15, yawRad: -0.5, pitchRad: -0.06, sigmaS: 0.27 },
    { t: 24.5, yawRad: 0.45, pitchRad: 0.04, sigmaS: 0.2 },
    { t: 33, yawRad: -0.6, pitchRad: 0, sigmaS: 0.3 },
  ],
};

/** Artefacts shared by the '*-real' presets (iPhone video capture). */
const REAL_CAPTURE: Omit<SynthConfigInput, 'scene'> = {
  rsReadoutMs: 22,
  // adaptive: steps ≤ 0.3 px between bands (≈ 14 bands at 0.3 rad/s, up to 48 during fast pans)
  rsBands: 48,
  rsBandStepPx: 0.3,
  rsReverse: false,
  focusBreathing: { ...DEFAULT_FOCUS_BREATHING },
  exposureJitter: 0.15,
  whiteBalanceDrift: 0.03,
  jpegQuality: 0.85,
  // adaptive motion blur: 1 + ⌈motion / 1.5 px⌉ sub-frames per band, at most 12
  effects: { motionBlurSamples: 12, motionBlurPxPerSample: 1.5 },
};

/**
 * Named presets. 'object-real' (portrait θ = 0, 8 fps × 25 s) and 'yard-real' (landscape θ = 90,
 * 6 fps × 40 s): the clean scenes plus every real-capture artefact (rolling shutter 22 ms,
 * focus breathing ±0.6 %, tremor + fast pans, adaptive motion blur, ±15 % exposure drift,
 * ±3 % white-balance drift, JPEG q = 0.85).
 */
export const SYNTH_PRESETS: Record<SynthPresetName, SynthConfigInput> = {
  'object-real': { scene: 'object', ...REAL_CAPTURE, trajectory: { shake: OBJECT_REAL_SHAKE } },
  'yard-real': { scene: 'yard', ...REAL_CAPTURE, trajectory: { shake: YARD_REAL_SHAKE } },
};

export function isSynthPreset(name: string): name is SynthPresetName {
  return Object.prototype.hasOwnProperty.call(SYNTH_PRESETS, name);
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/** Deep merge of plain objects (arrays and null replace; undefined keeps the base). */
export function mergeConfigInput<T>(base: T, over: unknown): T {
  if (!isPlainObject(base) || !isPlainObject(over)) return (over === undefined ? base : over) as T;
  const out: Record<string, unknown> = { ...base };
  for (const [k, v] of Object.entries(over)) {
    if (v === undefined) continue;
    out[k] = isPlainObject(v) && isPlainObject(out[k]) ? mergeConfigInput(out[k], v) : v;
  }
  return out as T;
}

export function resolveConfig(raw: SynthConfigInput): SynthConfig {
  let input = raw;
  if (raw.preset) {
    if (!isSynthPreset(raw.preset)) throw new Error(`unknown synth preset '${raw.preset}'`);
    const p = SYNTH_PRESETS[raw.preset];
    if (raw.scene && raw.scene !== p.scene) throw new Error(`preset '${raw.preset}' is a '${p.scene}' scene, got scene '${raw.scene}'`);
    input = mergeConfigInput(structuredClone(p), raw);
    input.name = raw.name ?? raw.preset;
  }
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
  const fbIn = input.focusBreathing;
  return {
    scene: input.scene,
    preset: input.preset ?? null,
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
    rsReadoutMs: input.rsReadoutMs ?? 0,
    rsBands: input.rsBands ?? 32,
    rsBandStepPx: input.rsBandStepPx ?? 0,
    rsReverse: input.rsReverse ?? false,
    focusBreathing: fbIn ? { ...DEFAULT_FOCUS_BREATHING, ...fbIn } : null,
    whiteBalanceDrift: input.whiteBalanceDrift ?? 0,
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

/** Trajectory time (s) of frame i (= mid-readout and mid-exposure time of the frame). */
export function frameTime(cfg: SynthConfig, i: number): number {
  return i / cfg.fps;
}

/** True when any real-capture artefact beyond the legacy effects is on (extra frames.json fields). */
export function hasCaptureArtefacts(cfg: SynthConfig): boolean {
  const sh = cfg.trajectory.shake;
  return (
    cfg.rsReadoutMs > 0 ||
    cfg.focusBreathing !== null ||
    cfg.whiteBalanceDrift > 0 ||
    cfg.effects.motionBlurPxPerSample > 0 ||
    !!(sh && ((sh.tremorDeg ?? 0) !== 0 || (sh.tremorM ?? 0) !== 0 || (sh.pans?.length ?? 0) > 0))
  );
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
