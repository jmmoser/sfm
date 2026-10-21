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
 *
 * Not modelled: specular materials, lighting changes beyond the exposure drift, a camera–IMU
 * lever arm. Global shutter, fixed focal and a smooth path unless a `*-real` preset (or the
 * realism options) adds rolling shutter, focus breathing, shake / fast pans and white-balance
 * drift. SwiftShader renders ≈ 0.75–0.9 s per frame (a 240-frame yard dataset ≈ 4 min).
 */
import type { CameraIntrinsics, Pose } from '../../core/types';
import {
  exposureGainFn,
  frameTime,
  hasCaptureArtefacts,
  intrinsicsFor,
  resolveConfig,
  screenAngleFor,
  trajectoryFor,
  type SynthConfig,
  type SynthConfigInput,
} from './config';
import { cameraCenter, rCamDevice } from './conventions';
import {
  blurSampleCount,
  centerFocusDistance,
  focusBreathingSeries,
  globalShutterBand,
  imageMotionPx,
  readoutBands,
  readoutGeometry,
  rectProbes,
  rsBandCount,
  subframeOffsetsMs,
  whiteBalanceFn,
  type ReadoutBand,
} from './realism';
import { simulateGps } from './gps';
import { simulateHeadings } from './heading';
import { simulateImu } from './imu';
import { SynthRenderer, type RenderBand } from './render';
import { hashString, SynthRng } from './rng';
import { accumulateVisibility, quantizeNormals, sampleSurfaces } from './sampling';
import type { SynthScene } from './scene-common';
import { buildObjectScene } from './scene-object';
import { buildYardScene } from './scene-yard';
import { angularSpeedStats, panPeakRate, trajectoryPose } from './trajectory';

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
  const fb = cfg.focusBreathing;
  const renderer = new SynthRenderer(synth, cam, cfg.effects, fb ? { minFocal: cam.f * (1 - Math.abs(fb.amplitude)) } : {});
  sink.log(`[synth] ${cfg.name}: ${N} frames ${cam.width}x${cam.height}, f=${cam.f.toFixed(2)}, RT ${renderer.renderTargetSize.join('x')}, scene ${timings.scene.toFixed(0)} ms`);

  // ---- real-capture artefacts (all off by default; see realism.ts) ----
  const artefacts = hasCaptureArtefacts(cfg);
  const adaptiveBlur = cfg.effects.motionBlurPxPerSample > 0;
  const geom = cfg.rsReadoutMs > 0 ? readoutGeometry(theta, cam.width, cam.height, cfg.rsReverse) : null;
  const useBands = geom !== null || adaptiveBlur;
  const fullRect = { u0: 0, u1: cam.width - 1, v0: 0, v1: cam.height - 1 };
  const wbOf = whiteBalanceFn(cfg.seed, cfg.whiteBalanceDrift);
  // Autofocus target distance per frame (centre depth at the GT pose): drives focus breathing
  // and the depth of the image-motion probes of the adaptive blur / motion statistics.
  let focusM: number[] | null = null;
  let focal: number[] | null = null;
  if (artefacts) {
    ts = performance.now();
    focusM = [];
    for (let i = 0; i < N; i++) focusM.push(centerFocusDistance(renderer.renderWindowDepth(trajectoryPose(traj, frameTime(cfg, i)), cfg.visibilityScale), cam, fb ?? undefined));
    timings.focus = performance.now() - ts;
    if (fb) {
      const series = focusBreathingSeries(focusM, Array.from({ length: N }, (_, i) => frameTime(cfg, i)), fb);
      focal = series.b.map((b) => cam.f * (1 + b));
    }
  }
  const frameStats: { omega: number; blurPx: number; rsPx: number; renders: number; blurMax: number; bands: number }[] = [];

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
      const camI: CameraIntrinsics = focal ? { ...cam, f: focal[i] } : cam;
      if (focal) renderer.setFocal(focal[i]);
      const wb = wbOf(tau);
      if (cfg.whiteBalanceDrift > 0) renderer.setWhiteBalance(wb);
      const gain = gainOf(tau, i);
      let a = performance.now();
      const seed = hashString(`frame${i}`, cfg.seed);
      const expMs = cfg.effects.exposureMs;
      const nbMax = Math.max(1, Math.round(cfg.effects.motionBlurSamples));
      let renders = 0,
        blurMax = 1;
      // Image motion (px, at the AF depth) over the exposure and over the readout, at the frame time.
      const depthProbe = focusM ? focusM[i] : 1;
      const fullProbes = rectProbes(fullRect);
      const blurPx = artefacts ? imageMotionPx(camI, trajectoryPose(traj, tau - expMs / 2000), trajectoryPose(traj, tau + expMs / 2000), fullProbes, depthProbe) : 0;
      const rsPx = cfg.rsReadoutMs > 0 ? imageMotionPx(camI, trajectoryPose(traj, tau - cfg.rsReadoutMs / 2000), trajectoryPose(traj, tau + cfg.rsReadoutMs / 2000), fullProbes, depthProbe) : 0;
      const bandsDef: ReadoutBand[] = geom
        ? readoutBands(geom, cam.width, cam.height, rsBandCount(rsPx, blurPx, cfg.rsBands, cfg.rsBandStepPx), cfg.rsReadoutMs)
        : [globalShutterBand(cam.width, cam.height)];
      if (useBands) {
        // Rolling shutter / adaptive blur: per band, sub-frames across the band's exposure,
        // centred on the band's readout time tau + dt_k.
        const bands: RenderBand[] = bandsDef.map((b) => {
          const tb = tau + b.dtMs / 1000;
          let n = nbMax;
          if (adaptiveBlur) {
            const p0 = trajectoryPose(traj, tb - expMs / 2000),
              p1 = trajectoryPose(traj, tb + expMs / 2000);
            n = blurSampleCount(imageMotionPx(camI, p0, p1, rectProbes(b.rect), depthProbe), cfg.effects.motionBlurPxPerSample, nbMax);
          }
          blurMax = Math.max(blurMax, n);
          const offs = subframeOffsetsMs(n, expMs, adaptiveBlur ? 'midpoints' : 'endpoints');
          return { rect: b.rect, poses: offs.map((o) => (o === 0 && b.dtMs === 0 ? pose : trajectoryPose(traj, tb + o / 1000))) };
        });
        renders = renderer.renderFrameBands(bands, gain, seed);
      } else {
        const poses: Pose[] = [];
        for (let k = 0; k < nbMax; k++) {
          const dt = nbMax === 1 ? 0 : (k / (nbMax - 1) - 0.5) * (expMs / 1000);
          poses.push(nbMax === 1 ? pose : trajectoryPose(traj, tau + dt));
        }
        blurMax = nbMax;
        renders = poses.length;
        renderer.renderFrame(poses, gain, seed);
      }
      if (artefacts) {
        const w = traj.evaluate(tau).omegaDev;
        frameStats.push({ omega: Math.hypot(w[0], w[1], w[2]), blurPx, rsPx, renders, blurMax, bands: bandsDef.length });
      }
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
        accumulateVisibility(gt.positions, visibility, pose, camI, wd);
        if (focus && focusVisibility) accumulateVisibility(focus.positions, focusVisibility, pose, camI, wd);
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
        ...(artefacts
          ? {
              f: camI.f,
              focusDistance: focusM ? focusM[i] : null,
              wb: cfg.whiteBalanceDrift > 0 ? wb : [1, 1, 1],
              omega: frameStats[i].omega,
              blurPx: frameStats[i].blurPx,
              rsPx: frameStats[i].rsPx,
              blurSamples: frameStats[i].blurMax,
              rsBands: frameStats[i].bands,
              renders: frameStats[i].renders,
            }
          : {}),
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

  function captureArtefactsMeta() {
    const stat = (xs: number[]) => {
      const v = [...xs].sort((a, b) => a - b);
      const q = (p: number) => (v.length ? v[Math.min(v.length - 1, Math.floor(p * v.length))] : null);
      return { p50: q(0.5), p90: q(0.9), max: v.length ? v[v.length - 1] : null };
    };
    const sh = cfg.trajectory.shake ?? null;
    const fs = focal ?? [];
    return {
      rollingShutter:
        geom && cfg.rsReadoutMs > 0
          ? {
              readoutMs: cfg.rsReadoutMs,
              bands: cfg.rsBands,
              bandStepPx: cfg.rsBandStepPx,
              axis: geom.axis,
              lines: geom.lines,
              lineDelayUs: (1000 * cfg.rsReadoutMs) / Math.max(1, geom.lines - 1),
              firstLine: geom.axis === 'cols' ? (geom.sign === 1 ? 'left column (u = 0)' : 'right column (u = W-1)') : geom.sign === 1 ? 'top row (v = 0)' : 'bottom row (v = H-1)',
              reverse: cfg.rsReverse,
              definition:
                'line i (readout order) of L is exposed around t_frame + (i/(L-1) - 0.5) * readoutMs; frames.json t/tTrue and pose are at mid-readout (s = 0.5). Native sensor rows (upright at screen angle 90) are read top to bottom, i.e. from the device +x edge to -x.',
            }
          : null,
      focusBreathing: fb
        ? { ...fb, f0: cam.f, fMin: Math.min(...fs), fMax: Math.max(...fs), fMean: fs.reduce((a, b) => a + b, 0) / Math.max(1, fs.length), note: 'per-frame GT focal in frames.json f; meta.intrinsics.f = f0' }
        : null,
      whiteBalanceDrift: cfg.whiteBalanceDrift,
      exposureJitter: cfg.exposureJitter,
      adaptiveBlur: adaptiveBlur ? { pxPerSample: cfg.effects.motionBlurPxPerSample, maxSamples: cfg.effects.motionBlurSamples, exposureMs: cfg.effects.exposureMs } : null,
      shake: sh ? { ...sh, panPeakRates: (sh.pans ?? []).map(panPeakRate) } : null,
      angularSpeed: angularSpeedStats(traj, 0, lastTau),
      frames: {
        omega: stat(frameStats.map((x) => x.omega)),
        blurPx: stat(frameStats.map((x) => x.blurPx)),
        rsPx: stat(frameStats.map((x) => x.rsPx)),
        renders: stat(frameStats.map((x) => x.renders)),
        rsBands: stat(frameStats.map((x) => x.bands)),
        focusDistance: focusM ? stat(focusM) : null,
      },
    };
  }

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
    preset: cfg.preset,
    captureArtefacts: artefacts ? captureArtefactsMeta() : null,
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
