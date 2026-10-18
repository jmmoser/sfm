/**
 * TEST-ONLY: synthetic hand-held captures for engine tests (not exported from ./index).
 *
 * A textured object scene (src/dense/synthetic ray caster, high-contrast materials so a
 * 240×320 frame yields ~900 SIFT features) is orbited by a camera with a hand-held vertical bob
 * and sway; frames are rendered upright for a UI angle θ (R = R_cam_device(θ)·R_wdᵀ), and an
 * exact IMU (gyro = ω_dev, accelG = R_wdᵀ(a − g), 60 Hz) is sampled from the same trajectory.
 * Frames report t = tTrue + timeOffsetMs (IMU on the true clock), like the e2e datasets.
 */
import type { CameraIntrinsics, ImuSample, Mat3, Pose, RGBAImage } from '../core/types';
import { lookAt, pinholeIntrinsics, renderSynthetic, type SynthScene } from '../dense/synthetic';
import { rotateRGBA90 } from '../features/image';
import { mat3TransposeMul, mat3Transpose } from '../math/mat3';
import { mat3ToRotvec } from '../math/rotation';
import { umeyama } from '../math/sim3';
import { cameraFromDevice } from '../sensors/frame-orientation';
import type { FrameInput, ScreenAngle } from '../workers/protocol';
import type { SparseResultEx } from './types';

export function objectScene(): SynthScene {
  const c = 6;
  return {
    primitives: [
      { kind: 'quad', center: [0, 0, 0], axisU: [1, 0, 0], axisV: [0, 1, 0], halfU: 4, halfV: 4, material: { scale: 0.25, octaves: 9, contrast: c, color: [0.62, 0.55, 0.42], seed: 1 } },
      { kind: 'box', center: [0.25, 0.15, 0.25], halfSize: [0.25, 0.2, 0.25], yaw: 0.4, material: { scale: 0.1, octaves: 8, contrast: c, color: [0.8, 0.42, 0.33], seed: 2 } },
      { kind: 'box', center: [-0.35, 0.35, 0.15], halfSize: [0.15, 0.15, 0.15], yaw: -0.3, material: { scale: 0.075, octaves: 8, contrast: c, color: [0.38, 0.55, 0.8], seed: 3 } },
      { kind: 'sphere', center: [-0.2, -0.35, 0.22], radius: 0.22, material: { scale: 0.075, octaves: 8, contrast: c, color: [0.45, 0.75, 0.4], seed: 4 } },
      { kind: 'quad', center: [0, 2.5, 1.2], axisU: [1, 0, 0], axisV: [0, 0, 1], halfU: 4, halfV: 1.2, material: { scale: 0.3, octaves: 9, contrast: c, color: [0.7, 0.7, 0.65], seed: 5 } },
    ],
  };
}

export interface SynthCaptureOptions {
  frames: number;
  fps: number;
  /** Rendered (upright) image size for θ. */
  width: number;
  height: number;
  hfovDeg: number;
  /** UI angle the frames are upright for. */
  theta: ScreenAngle;
  radius: number;
  camHeight: number;
  target: [number, number, number];
  startDeg: number;
  arcDeg: number;
  bobM: number;
  bobHz: number;
  swayM: number;
  imuHz: number;
  timeOffsetMs: number;
  t0: number;
  supersample: number;
  /** Render the frames (false: IMU / poses only, images are 1×1). */
  render: boolean;
}

export const DEFAULT_CAPTURE: SynthCaptureOptions = {
  frames: 60,
  fps: 10,
  width: 240,
  height: 320,
  hfovDeg: 55,
  theta: 0,
  radius: 1.4,
  camHeight: 0.9,
  target: [0, 0, 0.2],
  startDeg: -60,
  arcDeg: 75,
  bobM: 0.015,
  bobHz: 1.2,
  swayM: 0.01,
  imuHz: 60,
  timeOffsetMs: 25,
  t0: 10_000,
  supersample: 1,
  render: true,
};

export interface SynthFrame {
  index: number;
  /** Reported capture time (ms). */
  t: number;
  tTrue: number;
  image: RGBAImage;
  /** GT world-to-camera pose of the rendered (upright-for-θ) image. */
  pose: Pose;
}

export interface SynthCapture {
  opts: SynthCaptureOptions;
  cam: CameraIntrinsics;
  frames: SynthFrame[];
  imu: ImuSample[];
  /** True camera ← device rotation of the rendered images. */
  R_cam_device: Mat3;
  poseAt(tTrueMs: number): Pose;
}

function centreAt(o: SynthCaptureOptions, tS: number): number[] {
  const T = o.frames / o.fps;
  const phi = ((o.startDeg + (o.arcDeg * tS) / T) * Math.PI) / 180;
  const bob = o.bobM * Math.sin(2 * Math.PI * o.bobHz * tS);
  const sway = o.swayM * Math.sin(2 * Math.PI * 0.7 * o.bobHz * tS + 0.4);
  return [o.target[0] + (o.radius + sway) * Math.cos(phi), o.target[1] + (o.radius + sway) * Math.sin(phi), o.camHeight + bob];
}

function poseAtS(o: SynthCaptureOptions, tS: number): Pose {
  const C = centreAt(o, tS);
  // slowly wandering aim point
  const aim = [o.target[0] + 0.04 * Math.sin(0.9 * tS), o.target[1] + 0.04 * Math.cos(0.7 * tS), o.target[2] + 0.03 * Math.sin(1.3 * tS)];
  // a level camera (y down) is upright for any UI angle; θ only changes the device attitude
  // R_wd = Rᵀ · R_cd(θ) (and the image aspect)
  return lookAt(C, aim);
}

/** Device→world rotation of the device at time t for frames upright for θ: R_wd = Rᵀ · R_cd(θ). */
function rwdAt(o: SynthCaptureOptions, tS: number): Mat3 {
  return mat3TransposeMul(poseAtS(o, tS).R, cameraFromDevice(o.theta));
}

/** Render a capture (deterministic). */
export function makeCapture(partial: Partial<SynthCaptureOptions> = {}, scene: SynthScene = objectScene()): SynthCapture {
  const o: SynthCaptureOptions = { ...DEFAULT_CAPTURE, ...partial };
  const cam = pinholeIntrinsics(o.width, o.height, (o.hfovDeg * Math.PI) / 180);
  const frames: SynthFrame[] = [];
  for (let i = 0; i < o.frames; i++) {
    const tS = i / o.fps;
    const pose = poseAtS(o, tS);
    const image = o.render ? renderSynthetic(scene, cam, pose, { supersample: o.supersample }).image : { width: 1, height: 1, data: new Uint8ClampedArray(4) };
    const tTrue = o.t0 + 1000 * tS;
    frames.push({ index: i, t: tTrue + o.timeOffsetMs, tTrue, image, pose });
  }
  // IMU from 0.5 s before to 0.5 s after the frames
  const imu: ImuSample[] = [];
  const h = 0.002;
  const T = o.frames / o.fps;
  const dt = 1 / o.imuHz;
  for (let tS = -0.5; tS <= T + 0.5; tS += dt) {
    const Ra = rwdAt(o, tS - h);
    const Rb = rwdAt(o, tS + h);
    const R0 = rwdAt(o, tS);
    const w = mat3ToRotvec(mat3TransposeMul(Ra, Rb));
    const gyro: [number, number, number] = [w[0] / (2 * h), w[1] / (2 * h), w[2] / (2 * h)];
    const Cm = centreAt(o, tS - h);
    const C0 = centreAt(o, tS);
    const Cp = centreAt(o, tS + h);
    const a = [0, 1, 2].map((k) => (Cp[k] - 2 * C0[k] + Cm[k]) / (h * h));
    const RT = mat3Transpose(R0);
    const mul = (v: number[]): [number, number, number] => [
      RT[0] * v[0] + RT[1] * v[1] + RT[2] * v[2],
      RT[3] * v[0] + RT[4] * v[1] + RT[5] * v[2],
      RT[6] * v[0] + RT[7] * v[1] + RT[8] * v[2],
    ];
    imu.push({ t: o.t0 + 1000 * tS, gyro, accel: mul(a), accelG: mul([a[0], a[1], a[2] + 9.81]) });
  }
  return {
    opts: o,
    cam,
    frames,
    imu,
    R_cam_device: cameraFromDevice(o.theta),
    poseAt: (tTrueMs: number) => poseAtS(o, (tTrueMs - o.t0) / 1000),
  };
}

/**
 * FrameInputs of a capture: IMU samples since the previous frame (by the IMU clock ≈ frame
 * time), the reported screen angle and optionally the bitmap rotated by `rotate` clockwise
 * quarter turns (e.g. to simulate raw sensor buffers).
 */
export function frameInputs(
  cap: SynthCapture,
  opts: { screenAngle?: ScreenAngle; rotate?: number; from?: number; to?: number; copy?: boolean } = {},
): FrameInput[] {
  const out: FrameInput[] = [];
  const from = opts.from ?? 0;
  const to = opts.to ?? cap.frames.length;
  let k = 0;
  let lastT = -Infinity;
  if (from > 0) lastT = cap.frames[from - 1].tTrue + 30;
  for (let i = from; i < to; i++) {
    const f = cap.frames[i];
    const until = f.tTrue + 30; // the pump drains samples up to slightly past the frame
    const imu: ImuSample[] = [];
    while (k < cap.imu.length && cap.imu[k].t <= until) {
      if (cap.imu[k].t > lastT) imu.push(cap.imu[k]);
      k++;
    }
    lastT = until;
    let img = f.image;
    if (opts.rotate) img = rotateRGBA90(img, opts.rotate);
    else if (opts.copy !== false) img = { width: img.width, height: img.height, data: img.data.slice() };
    out.push({ bitmap: img, t: f.t, screenAngle: opts.screenAngle ?? cap.opts.theta, imu, index: f.index });
  }
  return out;
}

/** Camera centre C = −Rᵀt. */
export function centre(p: Pose): Float64Array {
  const R = p.R,
    t = p.t;
  return Float64Array.of(-(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]), -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]), -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]));
}

/** Sim3 (result → GT) from camera centres, ATE (m), max gravity error (deg). */
export function compareWithGT(cap: SynthCapture, res: SparseResultEx): { s: number; R: Float64Array; t: Float64Array; ate: number; gravityDeg: number } {
  const n = res.cameras.length;
  const src = new Float64Array(3 * n);
  const dst = new Float64Array(3 * n);
  let g = 0;
  res.cameras.forEach((c, i) => {
    const gt = cap.frames[c.frameIndex!].pose;
    src.set(centre(c.pose), 3 * i);
    dst.set(centre(gt), 3 * i);
    // world +Z expressed in the camera frame: result vs GT
    const d = c.pose.R[2] * gt.R[2] + c.pose.R[5] * gt.R[5] + c.pose.R[8] * gt.R[8];
    g = Math.max(g, Math.acos(Math.min(1, d)) / (Math.PI / 180));
  });
  const sim = umeyama(src, dst, n, true)!;
  let e = 0;
  for (let i = 0; i < n; i++) {
    for (let r = 0; r < 3; r++) {
      const p = sim.s * (sim.R[3 * r] * src[3 * i] + sim.R[3 * r + 1] * src[3 * i + 1] + sim.R[3 * r + 2] * src[3 * i + 2]) + sim.t[r];
      e += (p - dst[3 * i + r]) ** 2;
    }
  }
  return { s: sim.s, R: sim.R, t: sim.t, ate: Math.sqrt(e / n), gravityDeg: g };
}
