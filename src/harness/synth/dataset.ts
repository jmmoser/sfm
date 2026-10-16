/**
 * Typed loader for synthetic datasets written by e2e/generate-synth.mjs (browser or
 * Node ≥ 18 — uses fetch). The Vite dev server serves datasets at
 * `/e2e/data/<name>/` because e2e/data lives under the project root.
 */
import type { CameraIntrinsics, GeoFix, ImuSample, Pose } from '../../core/types';
import type { HeadingSample } from './heading';

export interface SynthBox {
  min: [number, number, number];
  max: [number, number, number];
}

export interface SynthMeta {
  name: string;
  scene: 'object' | 'yard';
  width: number;
  height: number;
  fps: number;
  numFrames: number;
  duration: number;
  startTimeMs: number;
  intrinsics: { f: number; cx: number; cy: number; k1: number; k2: number; width: number; height: number };
  screenAngle: 0 | 90 | 180 | 270;
  /** Row-major 3x3, v_cam = R_cam_device v_device (same for every frame). */
  R_cam_device: number[];
  originLatLonAlt: [number, number, number] | null;
  /** Diagonal (m) of the bounding box of all camera centres and the focus region. */
  sceneExtent: number;
  roi: SynthBox;
  focus: SynthBox;
  /** Default dense-evaluation threshold (m). */
  denseTau: number;
  /** Reported frame t = true t + timeOffsetMs (IMU/GPS/heading are on the true clock). */
  timeOffsetMs: number;
  imuNoise: { rateHz: number; timeJitterMs?: number; gyroNoise: number; accelNoise: number; gyroBias: number[]; accelBias: number[] };
  gpsNoise: { rateHz: number; horizontalSigma: number; verticalSigma: number; correlationS: number } | null;
  headingNoise: { rateHz: number; sigmaDeg: number; biasDeg: number };
  gt: {
    points: number;
    visible: number | null;
    totalArea: number;
    spacing: number;
    pointsFile: string;
    normalsFile: string;
    visibilityFile: string | null;
    focus: { points: number; visible: number | null; totalArea: number; spacing: number; pointsFile: string; normalsFile: string; visibilityFile: string | null } | null;
  };
  [k: string]: unknown;
}

export interface SynthFrame {
  index: number;
  /** Relative path of the JPEG inside the dataset directory. */
  file: string;
  /** Reported capture time (ms) = tTrue + timeOffsetMs. */
  t: number;
  /** Ground-truth time on the IMU clock (not available to a real pipeline). */
  tTrue: number;
  /** Ground-truth world-to-camera pose (row-major R, t). */
  pose: { R: number[]; t: number[] };
  center: [number, number, number];
  gain: number;
  depthFile?: string;
}

export interface SynthGeoFix extends GeoFix {
  trueEnu: [number, number, number];
}

export interface SynthDataset {
  baseUrl: string;
  meta: SynthMeta;
  frames: SynthFrame[];
  imu: ImuSample[];
  headings: HeadingSample[];
  geo: SynthGeoFix[] | null;
  /** URL of frame i's JPEG. */
  frameUrl(i: number): string;
  intrinsics(): CameraIntrinsics;
  /** Ground-truth pose of frame i as typed arrays. */
  pose(i: number): Pose;
}

type FetchLike = (url: string) => Promise<{ ok: boolean; status: number; json(): Promise<unknown>; arrayBuffer(): Promise<ArrayBuffer> }>;

function join(base: string, rel: string): string {
  return base.endsWith('/') ? base + rel : `${base}/${rel}`;
}

export async function loadSynthDataset(baseUrl: string, fetchFn: FetchLike = (u) => fetch(u)): Promise<SynthDataset> {
  const get = async <T>(rel: string, optional = false): Promise<T | null> => {
    const r = await fetchFn(join(baseUrl, rel));
    if (!r.ok) {
      if (optional) return null;
      throw new Error(`failed to load ${rel}: HTTP ${r.status}`);
    }
    return (await r.json()) as T;
  };
  const [meta, frames, imu, headings, geo] = await Promise.all([
    get<SynthMeta>('meta.json'),
    get<SynthFrame[]>('frames.json'),
    get<ImuSample[]>('imu.json'),
    get<HeadingSample[]>('heading.json'),
    get<SynthGeoFix[]>('geo.json', true),
  ]);
  const m = meta!;
  const fr = frames!;
  return {
    baseUrl,
    meta: m,
    frames: fr,
    imu: imu!,
    headings: headings!,
    geo,
    frameUrl: (i) => join(baseUrl, fr[i].file),
    intrinsics: () => ({ ...m.intrinsics }),
    pose: (i) => ({ R: Float64Array.from(fr[i].pose.R), t: Float64Array.from(fr[i].pose.t) }),
  };
}

/** Float32 binary file → typed array (GT points, depth maps). */
export async function loadFloat32(url: string, fetchFn: FetchLike = (u) => fetch(u)): Promise<Float32Array> {
  const r = await fetchFn(url);
  if (!r.ok) throw new Error(`failed to load ${url}: HTTP ${r.status}`);
  return new Float32Array(await r.arrayBuffer());
}
