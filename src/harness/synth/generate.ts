/**
 * Dataset generation: renders frames + GT depth, simulates IMU / GPS / compass,
 * samples GT surface points with visibility, and streams everything to a sink
 * (the Node generator exposes one through Playwright; see e2e/generate-synth.mjs).
 *
 * Output files (relative to the dataset directory):
 *   meta.json, frames.json, imu.json, heading.json, [geo.json], frames/NNNNNN.jpg,
 *   [depth/NNNNNN.f32], gt_points.bin (Float32 xyz), gt_normals.i8 (int8 xyz ×127),
 *   [gt_visibility.u8] (frames seeing each sample, saturating at 255), and the same
 *   three files with the gt_focus_ prefix for the dense object-of-interest samples.
 */
import type { CameraIntrinsics, Pose } from '../../core/types';
import {
  exposureGainFn,
  frameTime,
  intrinsicsFor,
  resolveConfig,
  screenAngleFor,
  trajectoryFor,
  type SynthConfig,
  type SynthConfigInput,
} from './config';
import { cameraCenter, rCamDevice } from './conventions';
import { simulateGps } from './gps';
import { simulateHeadings } from './heading';
import { simulateImu } from './imu';
import { SynthRenderer } from './render';
import { hashString, SynthRng } from './rng';
import { accumulateVisibility, quantizeNormals, sampleSurfaces } from './sampling';
import type { SynthScene } from './scene-common';
import { buildObjectScene } from './scene-object';
import { buildYardScene } from './scene-yard';
import { trajectoryPose } from './trajectory';

export interface SynthSink {
  /** Write (or append to) a file of the dataset. */
  write(path: string, data: Blob | ArrayBufferView | string, append?: boolean): Promise<void>;
  log(msg: string): void;
}

export interface SynthSummary {
  name: string;
  frames: number;
  width: number;
  height: number;
  gtPoints: number;
  visibleGtPoints: number;
  bytesWritten: number;
  timingsMs: Record<string, number>;
  renderTargetSize: [number, number];
}

export function buildScene(cfg: SynthConfig): SynthScene {
  const common = { seed: cfg.seed, anisotropy: cfg.anisotropy, textureScale: cfg.textureScale };
  return cfg.scene === 'object' ? buildObjectScene(common) : buildYardScene({ ...common, trajectory: cfg.trajectory });
}

function poseJSON(p: Pose): { R: number[]; t: number[] } {
  return { R: Array.from(p.R), t: Array.from(p.t) };
}

const pad6 = (i: number) => String(i).padStart(6, '0');

export async function generateDataset(input: SynthConfigInput, sink: SynthSink): Promise<SynthSummary> {
  const cfg = resolveConfig(input);
  const timings: Record<string, number> = {};
  const t0 = performance.now();
  let bytes = 0;
  const write = async (path: string, data: Blob | ArrayBufferView | string, append = false) => {
    bytes += typeof data === 'string' ? data.length : data instanceof Blob ? data.size : data.byteLength;
    await sink.write(path, data, append);
  };
  const writeChunked = async (path: string, data: ArrayBufferView) => {
    const u8 = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const CH = 6 << 20;
    if (u8.length === 0) await write(path, u8, false);
    for (let o = 0; o < u8.length; o += CH) await write(path, u8.subarray(o, Math.min(u8.length, o + CH)), o > 0);
  };

  const cam: CameraIntrinsics = intrinsicsFor(cfg);
  const theta = screenAngleFor(cfg);
  const traj = trajectoryFor(cfg);
  const N = cfg.frames;
  const lastTau = frameTime(cfg, N - 1);
  const T0 = cfg.startTimeMs;

  // ---- sensors (IMU clock = true time) ----
  let ts = performance.now();
  const imu = simulateImu(traj, -0.5, lastTau + 0.5, T0, cfg.imu);
  const headings = simulateHeadings(traj, -0.5, lastTau + 0.5, T0, cfg.heading);
  const geo = cfg.gps ? simulateGps(traj, -0.5, lastTau + 0.5, T0, cfg.gps) : null;
  timings.sensors = performance.now() - ts;

  // ---- scene + renderer ----
  ts = performance.now();
  const synth = buildScene(cfg);
  timings.scene = performance.now() - ts;
  const renderer = new SynthRenderer(synth, cam, cfg.effects);
  sink.log(`[synth] ${cfg.name}: ${N} frames ${cam.width}x${cam.height}, f=${cam.f.toFixed(2)}, RT ${renderer.renderTargetSize.join('x')}, scene ${timings.scene.toFixed(0)} ms`);

  // ---- GT surface samples ----
  ts = performance.now();
  const gt = sampleSurfaces(synth.surfaceMeshes, cfg.gtPoints, new SynthRng(cfg.seed).derive('gt-points'));
  const focus =
    cfg.gtFocusPoints > 0 && synth.focusMeshes.length > 0
      ? sampleSurfaces(synth.focusMeshes, cfg.gtFocusPoints, new SynthRng(cfg.seed).derive('gt-focus-points'))
      : null;
  const visibility = new Uint8Array(gt.count);
  const focusVisibility = focus ? new Uint8Array(focus.count) : null;
  timings.gtSampling = performance.now() - ts;

  const gainOf = exposureGainFn(cfg);
  const framesJson: unknown[] = [];
  const centers: number[][] = [];
  let tRender = 0,
    tEncode = 0,
    tDepth = 0,
    tVis = 0;
  try {
    for (let i = 0; i < N; i++) {
      const tau = frameTime(cfg, i);
      const pose = trajectoryPose(traj, tau);
      const nb = Math.max(1, Math.round(cfg.effects.motionBlurSamples));
      const poses: Pose[] = [];
      for (let k = 0; k < nb; k++) {
        const dt = nb === 1 ? 0 : (k / (nb - 1) - 0.5) * (cfg.effects.exposureMs / 1000);
        poses.push(nb === 1 ? pose : trajectoryPose(traj, tau + dt));
      }
      const gain = gainOf(tau, i);
      let a = performance.now();
      renderer.renderFrame(poses, gain, hashString(`frame${i}`, cfg.seed));
      const jpeg = await renderer.toJpeg(cfg.jpegQuality);
      const b2 = performance.now();
      tRender += b2 - a;
      const file = `frames/${pad6(i)}.jpg`;
      await write(file, jpeg);
      tEncode += performance.now() - b2;
      let depthFile: string | undefined;
      if (cfg.depthEvery > 0 && i % cfg.depthEvery === 0) {
        a = performance.now();
        const depth = renderer.renderDepth(pose);
        depthFile = `depth/${pad6(i)}.f32`;
        await write(depthFile, depth);
        tDepth += performance.now() - a;
      }
      if (cfg.visibilityEvery > 0 && i % cfg.visibilityEvery === 0) {
        a = performance.now();
        const wd = renderer.renderWindowDepth(pose, cfg.visibilityScale);
        accumulateVisibility(gt.positions, visibility, pose, cam, wd);
        if (focus && focusVisibility) accumulateVisibility(focus.positions, focusVisibility, pose, cam, wd);
        tVis += performance.now() - a;
      }
      const C = cameraCenter(pose);
      centers.push([C[0], C[1], C[2]]);
      framesJson.push({
        index: i,
        file,
        t: T0 + tau * 1000 + cfg.timeOffsetMs,
        tTrue: T0 + tau * 1000,
        pose: poseJSON(pose),
        center: [C[0], C[1], C[2]],
        gain,
        ...(depthFile ? { depthFile } : {}),
      });
      if (i % 10 === 0 || i === N - 1) sink.log(`[synth] ${cfg.name}: frame ${i + 1}/${N}`);
    }
  } finally {
    renderer.dispose();
    synth.dispose();
  }
  timings.render = tRender;
  timings.encodeWrite = tEncode;
  timings.depth = tDepth;
  timings.visibility = tVis;

  // ---- GT files ----
  ts = performance.now();
  await writeChunked('gt_points.bin', gt.positions);
  await writeChunked('gt_normals.i8', quantizeNormals(gt.normals));
  let visibleCount = 0,
    focusVisibleCount = 0;
  if (cfg.visibilityEvery > 0) {
    for (let i = 0; i < visibility.length; i++) if (visibility[i] > 0) visibleCount++;
    await writeChunked('gt_visibility.u8', visibility);
  }
  if (focus) {
    await writeChunked('gt_focus_points.bin', focus.positions);
    await writeChunked('gt_focus_normals.i8', quantizeNormals(focus.normals));
    if (cfg.visibilityEvery > 0 && focusVisibility) {
      for (let i = 0; i < focusVisibility.length; i++) if (focusVisibility[i] > 0) focusVisibleCount++;
      await writeChunked('gt_focus_visibility.u8', focusVisibility);
    }
  }
  timings.gtWrite = performance.now() - ts;

  // ---- scene extent: diagonal of bbox(camera centres ∪ focus box) ----
  const lo = [...synth.focus.min],
    hi = [...synth.focus.max];
  for (const c of centers)
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], c[k]);
      hi[k] = Math.max(hi[k], c[k]);
    }
  const sceneExtent = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);

  // ---- JSON ----
  await write('frames.json', JSON.stringify(framesJson));
  await write('imu.json', JSON.stringify(imu));
  await write('heading.json', JSON.stringify(headings));
  if (geo) await write('geo.json', JSON.stringify(geo));
  timings.total = performance.now() - t0;
  const meta = {
    name: cfg.name,
    scene: cfg.scene,
    generator: 'sfm synth v1',
    width: cam.width,
    height: cam.height,
    fps: cfg.fps,
    numFrames: N,
    duration: cfg.duration,
    startTimeMs: T0,
    intrinsics: { f: cam.f, cx: cam.cx, cy: cam.cy, k1: cam.k1, k2: cam.k2, width: cam.width, height: cam.height },
    cameraModel: 'RADIAL (pixel centres at integer coordinates)',
    screenAngle: theta,
    R_cam_device: Array.from(rCamDevice(theta)),
    worldFrame: 'ENU, Z-up, metres (x east, y north, z up)',
    gravityWorld: [0, 0, -9.81],
    originLatLonAlt: cfg.gps ? cfg.gps.originLatLonAlt : null,
    sceneExtent,
    sceneBounds: { min: lo, max: hi },
    roi: synth.roi,
    focus: synth.focus,
    denseTau: cfg.scene === 'object' ? 0.01 * sceneExtent : 0.1,
    timeOffsetMs: cfg.timeOffsetMs,
    timeOffsetConvention: 'frames.json t = true time (IMU clock) + timeOffsetMs; tTrue is the ground truth',
    imuNoise: {
      rateHz: cfg.imu.rateHz,
      timeJitterMs: cfg.imu.timeJitterMs,
      gyroNoise: cfg.imu.gyroNoise,
      accelNoise: cfg.imu.accelNoise,
      gyroBias: cfg.imu.gyroBias,
      accelBias: cfg.imu.accelBias,
      note: 'white noise sigma per sample; the same accelerometer noise/bias is applied to accel and accelG',
    },
    gpsNoise: cfg.gps
      ? { rateHz: cfg.gps.rateHz, horizontalSigma: cfg.gps.horizontalSigma, verticalSigma: cfg.gps.verticalSigma, correlationS: cfg.gps.correlationS }
      : null,
    headingNoise: { rateHz: cfg.heading.rateHz, sigmaDeg: cfg.heading.sigmaDeg, biasDeg: cfg.heading.biasDeg, definition: 'camera optical axis projected to horizontal, deg clockwise from true north' },
    render: { ...cfg.effects, exposureJitter: cfg.exposureJitter, jpegQuality: cfg.jpegQuality, renderTargetSize: renderer.renderTargetSize },
    depth: { every: cfg.depthEvery, format: 'Float32 little-endian, width*height, row 0 = top, z-depth (m) of the surface at each pixel centre of the distorted JPEG, 0 = none' },
    gt: {
      points: gt.count,
      visible: cfg.visibilityEvery > 0 ? visibleCount : null,
      totalArea: gt.totalArea,
      pointsFile: 'gt_points.bin',
      normalsFile: 'gt_normals.i8',
      visibilityFile: cfg.visibilityEvery > 0 ? 'gt_visibility.u8' : null,
      visibilityEvery: cfg.visibilityEvery,
      /** Mean spacing of the area-uniform samples, sqrt(area / count). */
      spacing: Math.sqrt(gt.totalArea / Math.max(1, gt.count)),
      focus: focus
        ? {
            points: focus.count,
            visible: cfg.visibilityEvery > 0 ? focusVisibleCount : null,
            totalArea: focus.totalArea,
            spacing: Math.sqrt(focus.totalArea / Math.max(1, focus.count)),
            pointsFile: 'gt_focus_points.bin',
            normalsFile: 'gt_focus_normals.i8',
            visibilityFile: cfg.visibilityEvery > 0 ? 'gt_focus_visibility.u8' : null,
          }
        : null,
    },
    seed: cfg.seed,
    config: cfg,
    timingsMs: timings,
  };
  await write('meta.json', JSON.stringify(meta, null, 2));
  sink.log(`[synth] ${cfg.name}: done in ${(timings.total / 1000).toFixed(1)} s (${(bytes / 1048576).toFixed(1)} MB)`);
  return {
    name: cfg.name,
    frames: N,
    width: cam.width,
    height: cam.height,
    gtPoints: gt.count,
    visibleGtPoints: visibleCount,
    bytesWritten: bytes,
    timingsMs: timings,
    renderTargetSize: renderer.renderTargetSize,
  };
}
