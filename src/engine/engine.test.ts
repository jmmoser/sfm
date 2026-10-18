/**
 * PhotogrammetryEngine on the CPU path (Node): synthetic hand-held orbits rendered with the
 * dense ray caster + exact IMU (src/engine/testutil.ts). Covers frame ingest (canonical
 * orientation, settle, raw sensor buffers), keyframing, live mapping, refine (registration,
 * self-calibration, world alignment with the simulated IMU: gravity, metric scale, time
 * offset), the R_cam_device hand-eye check, dense + mesh + exports, persistence / resume and
 * loadSession, visibility and cancellation.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import type { FeatureSet, Pose } from '../core/types';
import { pinholeIntrinsics, renderSynthetic } from '../dense/synthetic';
import type { FeaturePipeline } from '../features/extractor';
import { parseGLB } from '../io/glb';
import { decodePLY } from '../io/ply';
import { readZip } from '../io/zip';
import { umeyama } from '../math/sim3';
import { rotateImagePoint } from '../sensors/frame-orientation';
import type { LiveUpdate } from '../workers/protocol';
import { createCpuFeaturePipeline } from './cpu-pipeline';
import { PhotogrammetryEngine, frameToBitmap } from './engine';
import { OrientationTracker, gravityDown, overlayPoints, prepareFrame } from './frame-ingest';
import { ImuLog, classifyScene, handEyeCheck } from './live';
import { MemoryEngineStore, packImu, packSparseModel, unpackImu, unpackSparseModel } from './persistence';
import { centre, frameInputs, makeCapture, objectScene, type SynthCapture } from './testutil';
import type { EngineConfig, SparseResultEx } from './types';

const DEG = Math.PI / 180;

let pipeline: FeaturePipeline;
beforeAll(() => {
  pipeline = createCpuFeaturePipeline({ maxFeatures: 2048, maxImageSize: 1024 });
});

function testConfig(extra: EngineConfig = {}): EngineConfig {
  return { pipeline, mapping: 'inline', ...extra };
}

/** Sim3 (result → GT) from camera centres, ATE (m), max gravity error (deg). */
function compare(cap: SynthCapture, res: SparseResultEx): { s: number; R: Float64Array; t: Float64Array; ate: number; gravityDeg: number } {
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
    g = Math.max(g, Math.acos(Math.min(1, d)) / DEG);
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

// ---------------------------------------------------------------------------------------------

describe('frame ingest', () => {
  it('a declared raw frame is upright for θ = 90 in every UI rotation and never changes the delivery-path state', () => {
    // MSTP-free VideoFrame grab: landscape-90 → landscape-270 without a portrait frame in between
    const o = new OrientationTracker(500);
    expect(o.classify({ width: 640, height: 480, screenAngle: 90, t: 0, raw: true })).toMatchObject({ ok: true, effectiveTheta: 90, k: 0, raw: true });
    expect(o.thetaSession).toBe(90);
    expect(o.classify({ width: 640, height: 480, screenAngle: 270, t: 700, raw: true }).ok).toBe(false); // settling after the rotation
    expect(o.classify({ width: 640, height: 480, screenAngle: 270, t: 1300, raw: true })).toMatchObject({ ok: true, effectiveTheta: 90, k: 0 });
    o.classify({ width: 640, height: 480, screenAngle: 0, t: 1400, raw: true });
    expect(o.classify({ width: 640, height: 480, screenAngle: 0, t: 2000, raw: true })).toMatchObject({ ok: true, effectiveTheta: 90, k: 0 });
    expect(o.rawDelivery).toBeNull(); // per frame only: nothing persisted
    expect(o.classify({ width: 480, height: 640, screenAngle: 0, t: 2100, raw: true }).ok).toBe(false); // portrait "raw" frame
    // a session started in landscape-270 on the raw path is not ambiguous
    const s = new OrientationTracker(500);
    s.classify({ width: 640, height: 480, screenAngle: 270, t: 0, raw: true });
    expect(s.thetaSession).toBe(90);
    // resumed: captured with upright frames at θ₀ = 270 (rawDelivery false) → raw frames rotate in, state kept
    const r = new OrientationTracker(500, 270);
    r.rawDelivery = false;
    expect(r.classify({ width: 640, height: 480, screenAngle: 270, t: 0, raw: true })).toMatchObject({ ok: true, effectiveTheta: 90, k: 2 });
    expect(r.thetaSession).toBe(270);
    expect(r.rawDelivery).toBe(false);
    // ambiguous resumed session (θ₀ = 270, rawDelivery null): NOT re-interpreted by raw frames…
    const a = new OrientationTracker(500, 270);
    a.classify({ width: 640, height: 480, screenAngle: 270, t: 0, raw: true });
    expect(a.thetaSession).toBe(270);
    expect(a.rawDelivery).toBeNull();
    // …and an upright frame of the <video> path at 270 afterwards is still upright for 270
    expect(a.classify({ width: 640, height: 480, screenAngle: 270, t: 100 })).toMatchObject({ ok: true, effectiveTheta: 270, k: 0, raw: false });
  });

  it('frameToBitmap converts a VideoFrame input (downscaled to maxSide) and always closes it', async () => {
    const g = globalThis as { VideoFrame?: unknown; createImageBitmap?: unknown };
    const saved = { VideoFrame: g.VideoFrame, createImageBitmap: g.createImageBitmap };
    class FakeVideoFrame {
      closed = false;
      constructor(
        readonly displayWidth: number,
        readonly displayHeight: number,
      ) {}
      close(): void {
        this.closed = true;
      }
    }
    let fail = false;
    const calls: unknown[] = [];
    g.VideoFrame = FakeVideoFrame;
    g.createImageBitmap = (src: unknown, opts?: { resizeWidth?: number; resizeHeight?: number }) => {
      calls.push(opts);
      if (fail) return Promise.reject(new Error('decode failed'));
      return Promise.resolve({ width: opts?.resizeWidth ?? 1920, height: opts?.resizeHeight ?? 1440, close() {} });
    };
    try {
      const vf = new FakeVideoFrame(1920, 1440);
      const base = { t: 1, screenAngle: 0 as const, imu: [], rawBuffer: true };
      const out = await frameToBitmap({ ...base, bitmap: vf as unknown as VideoFrame, maxSide: 1280 });
      expect([(out.bitmap as ImageBitmap).width, (out.bitmap as ImageBitmap).height]).toEqual([1280, 960]);
      expect(out.rawBuffer).toBe(true);
      expect(vf.closed).toBe(true);
      const native = new FakeVideoFrame(1920, 1440);
      await frameToBitmap({ ...base, bitmap: native as unknown as VideoFrame });
      expect(calls[1]).toBeUndefined(); // no maxSide: native size
      fail = true;
      const bad = new FakeVideoFrame(1920, 1440);
      await expect(frameToBitmap({ ...base, bitmap: bad as unknown as VideoFrame })).rejects.toThrow(/decode failed/);
      expect(bad.closed).toBe(true); // the camera buffer is released on errors too
      // an ImageBitmap / RGBA input passes through unchanged
      const rgba = { width: 2, height: 2, data: new Uint8ClampedArray(16) };
      const same = { ...base, bitmap: rgba };
      expect(await frameToBitmap(same)).toBe(same);
    } finally {
      g.VideoFrame = saved.VideoFrame;
      g.createImageBitmap = saved.createImageBitmap;
    }
  });

  it('fixes the canonical θ, drops frames after a rotation and detects raw sensor buffers', () => {
    const o = new OrientationTracker(500);
    // first frame: portrait UI, portrait bitmap
    let v = o.classify({ width: 480, height: 640, screenAngle: 0, t: 0 });
    expect(v).toMatchObject({ ok: true, effectiveTheta: 0, k: 0, raw: false });
    expect(o.thetaSession).toBe(0);
    // landscape bitmap in a portrait UI = raw sensor buffer (upright for θ = 90) → 1 clockwise turn
    v = o.classify({ width: 640, height: 480, screenAngle: 0, t: 100 });
    expect(v).toMatchObject({ ok: true, effectiveTheta: 90, k: 1, raw: true });
    // UI rotates to landscape: frames within 500 ms are dropped
    v = o.classify({ width: 640, height: 480, screenAngle: 90, t: 200 });
    expect(v.ok).toBe(false);
    v = o.classify({ width: 640, height: 480, screenAngle: 90, t: 650 });
    expect(v.ok).toBe(false);
    v = o.classify({ width: 640, height: 480, screenAngle: 90, t: 750 });
    expect(v).toMatchObject({ ok: true, effectiveTheta: 90, k: 1 });
    // portrait bitmap in a landscape UI: transient, dropped
    v = o.classify({ width: 480, height: 640, screenAngle: 90, t: 800 });
    expect(v.ok).toBe(false);
    // 270 relative to the portrait session = 3 clockwise turns
    o.classify({ width: 640, height: 480, screenAngle: 270, t: 900 });
    v = o.classify({ width: 640, height: 480, screenAngle: 270, t: 1500 });
    expect(v).toMatchObject({ ok: true, k: 3 });
  });

  it('rotates RGBA frames into the canonical orientation and maps the overlay back', () => {
    const W = 40,
      H = 30;
    const data = new Uint8ClampedArray(W * H * 4).fill(30);
    // a bright pixel at (x=5, y=7)
    const at = (x: number, y: number): number => 4 * (y * W + x);
    data.set([250, 250, 250, 255], at(5, 7));
    const prep = prepareFrame({ width: W, height: H, data }, 1);
    expect([prep.width, prep.height]).toEqual([H, W]);
    const fi = prep.featureInput(false, 1000);
    expect(fi.kind).toBe('gray');
    const g = fi.kind === 'gray' ? fi.gray : null!;
    // clockwise turn: (x, y) → (H − 1 − y, x)
    const p = rotateImagePoint(5, 7, W, H, 1);
    let best = 0,
      bi = -1;
    for (let i = 0; i < g.data.length; i++)
      if (g.data[i] > best) {
        best = g.data[i];
        bi = i;
      }
    expect([bi % g.width, Math.floor(bi / g.width)]).toEqual([p[0], p[1]]);
    // overlay of a canonical keypoint, mapped back with the inverse turn (k = 3)
    const fs: FeatureSet = { count: 1, keypoints: Float32Array.of(p[0], p[1], 1, 0), descriptors: new Uint8Array(128), scores: Float32Array.of(1), width: H, height: W };
    const ov = overlayPoints(fs, prep.width, prep.height, 3, 10);
    expect(ov[0]).toBeCloseTo(5.5 / W, 6);
    expect(ov[1]).toBeCloseTo(7.5 / H, 6);
    prep.release();
  });

  it('gravity from the IMU (spec sign) and IMU log ordering / packing', () => {
    const log = new ImuLog();
    log.append([
      { t: 0, gyro: [0, 0, 0], accel: [0, 0, 0], accelG: [0, 9.81, 0] },
      { t: 20, gyro: [0, 0, 0], accel: null, accelG: [0, 9.81, 0] },
    ]);
    log.append([
      { t: 10, gyro: [0.1, 0, 0], accel: null, accelG: [0, 9.81, 0] },
      { t: 20, gyro: [9, 9, 9], accel: null, accelG: null },
    ]);
    expect(log.samples.map((s) => s.t)).toEqual([0, 10, 20]);
    // upright portrait: specific force along +y ⇒ gravity points down = −y
    expect(gravityDown(log.samples, 10)).toEqual([-0, -1, -0]);
    const back = unpackImu(packImu(log.samples).buffer as ArrayBuffer);
    expect(back.length).toBe(3);
    expect(back[1].gyro[0]).toBeCloseTo(0.1);
    expect(back[1].accel).toBeNull();
  });
});

// ---------------------------------------------------------------------------------------------

describe('engine: synthetic orbit, CPU path', () => {
  let cap: SynthCapture;
  let store: MemoryEngineStore;
  let engine: PhotogrammetryEngine;
  let res: SparseResultEx;
  const updates: LiveUpdate[] = [];

  // rendering the capture takes a few seconds; more on a loaded machine (default hook timeout 10 s)
  beforeAll(() => {
    cap = makeCapture();
    store = new MemoryEngineStore('orbit');
  }, 120_000);

  it('captures: keyframes, live mapping, overlay, live model updates', async () => {
    engine = new PhotogrammetryEngine({ mode: 'object', quality: 'fast', persist: false, sessionId: 'orbit', hfovPriorDeg: 60 }, testConfig({ store }));
    const init = await engine.init();
    expect(init.gpu).toBe(false);
    for (const f of frameInputs(cap)) updates.push(await engine.processFrame(f));
    const last = updates[updates.length - 1];
    const kfs = updates.filter((u) => u.decision === 'keyframe').length;
    expect(kfs).toBeGreaterThanOrEqual(12);
    expect(last.numKeyframes).toBe(kfs);
    expect(last.numRegistered).toBeGreaterThanOrEqual(0.9 * kfs);
    expect(last.numPoints).toBeGreaterThan(500);
    expect(last.tracking).toBe('tracking');
    expect(updates.some((u) => u.sparse && u.sparse.positions.length > 0 && u.cameras && u.cameras.length > 2)).toBe(true);
    // frame indices are the FrameInput indices; overlay in [0, 1]
    expect(updates.map((u) => u.frameIndex)).toEqual(cap.frames.map((f) => f.index));
    const ov = updates.find((u) => u.overlay && u.overlay.length > 0)!.overlay!;
    for (const v of ov) expect(v >= 0 && v <= 1).toBe(true);
    // the first frames warm up the sharpness statistic, guidance says initialising
    expect(updates[0].guidance).toBe('initializing');
    // R_cam_device verified against the gyro
    expect(engine.liveSession!.state.cameraThetaVerified).toBe(true);
    expect(engine.liveSession!.state.cameraTheta).toBe(0);
  }, 300_000);

  it('poses tracked frames in the live model (the camera view of the capture screen)', () => {
    const posed = updates.filter((u) => u.framePose);
    // most frames after initialisation are posed, not only keyframes
    expect(posed.length).toBeGreaterThan(0.5 * updates.filter((u) => u.tracking === 'tracking').length);
    expect(posed.length).toBeGreaterThan(updates.filter((u) => u.decision === 'keyframe').length);
    for (const u of posed) expect(u.R_cam_device).toBeDefined();
    // up to the live model's similarity: rotations and centres agree with the ground truth
    const n = posed.length;
    const src = new Float64Array(3 * n),
      dst = new Float64Array(3 * n);
    posed.forEach((u, i) => {
      src.set(centre(u.framePose!.pose), 3 * i);
      dst.set(centre(cap.frames.find((f) => f.index === u.frameIndex)!.pose), 3 * i);
    });
    const sim = umeyama(src, dst, n, true)!;
    let worst = 0,
      ate = 0,
      radius = 0;
    posed.forEach((u, i) => {
      const gt = cap.frames.find((f) => f.index === u.frameIndex)!.pose;
      // R_est · Rsimᵀ ≈ R_gt
      const Re = u.framePose!.pose.R;
      let tr = 0;
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
        let v = 0;
        for (let k = 0; k < 3; k++) v += Re[3 * r + k] * sim.R[3 * c + k];
        tr += v * gt.R[3 * r + c];
      }
      worst = Math.max(worst, Math.acos(Math.min(1, Math.max(-1, (tr - 1) / 2))) / DEG);
      for (let r = 0; r < 3; r++) {
        const p = sim.s * (sim.R[3 * r] * src[3 * i] + sim.R[3 * r + 1] * src[3 * i + 1] + sim.R[3 * r + 2] * src[3 * i + 2]) + sim.t[r];
        ate += (p - dst[3 * i + r]) ** 2;
      }
      radius = Math.max(radius, Math.hypot(dst[3 * i], dst[3 * i + 1], dst[3 * i + 2]));
    });
    expect(worst).toBeLessThan(1.5);
    expect(Math.sqrt(ate / n)).toBeLessThan(0.01 * radius);
  });

  it('refines: all keyframes registered, self-calibrated focal, metric gravity-aligned world', async () => {
    await engine.finishCapture();
    const stages = new Set<string>();
    res = await engine.refine((p) => stages.add(p.stage));
    expect([...stages]).toEqual(expect.arrayContaining(['matching', 'reconstruction', 'alignment']));
    const nk = engine.liveSession!.state.order.length;
    expect(res.cameras.length).toBe(nk);
    expect(res.stats.registeredImages).toBe(nk);
    // t / frameIndex map keyframes to dataset frames
    for (const c of res.cameras) {
      expect(c.frameIndex).toBeGreaterThanOrEqual(0);
      expect(c.t).toBe(cap.frames[c.frameIndex!].t);
    }
    const fTrue = cap.cam.f;
    expect(Math.abs(res.cameras[0].intrinsics.f / fTrue - 1)).toBeLessThan(0.03);
    expect(res.metric.metric).toBe(true);
    expect(res.metric.source).toBe('imu');
    expect(res.metric.gravityAligned).toBe(true);
    const cmp = compare(cap, res);
    expect(Math.abs(cmp.s - 1)).toBeLessThan(0.03); // metric scale within 3 %
    expect(cmp.ate).toBeLessThan(0.005); // m, on a 1.4 m orbit
    expect(cmp.gravityDeg).toBeLessThan(1);
    // camera–IMU offset (frames stamped +25 ms late ⇒ offset −25 ms)
    expect(Math.abs((res.stats.timeOffsetMs as number) + 25)).toBeLessThan(3);
    expect(res.points.count).toBe(res.stats.points);
  }, 300_000);

  it('densifies (CPU PatchMatch) into a world-frame cloud that matches the scene, then meshes', async () => {
    {
      // a keyframe whose photo never reached storage is skipped, not a hard failure
      const missing = res.cameras[0].id; // the first registered view is always in the dense subset
      const getRecords = store.getFrameRecords.bind(store),
        getImage = store.getFrameImage.bind(store);
      store.getFrameRecords = async () => (await getRecords()).filter((r) => r.id !== missing);
      store.getFrameImage = async (id) => (id === missing ? null : getImage(id));
      try {
        const pc = await engine.densify({ maxImageSize: 96, maxViews: 8 });
        expect(pc.count).toBeGreaterThan(500);
      } finally {
        store.getFrameRecords = getRecords;
        store.getFrameImage = getImage;
      }
    }
    const pc = await engine.densify({ maxImageSize: 96, maxViews: 8 });
    expect(pc.count).toBeGreaterThan(1500);
    // GT depth of frame 0 vs the dense points mapped into the GT frame
    const cmp = compare(cap, res);
    const gtCam = pinholeIntrinsics(120, 160, (55 * Math.PI) / 180);
    const pose0: Pose = cap.frames[0].pose;
    const gt = renderSynthetic(objectScene(), gtCam, pose0, { supersample: 1 });
    let n = 0,
      good = 0;
    for (let i = 0; i < pc.count; i += 3) {
      const x = [pc.positions[3 * i], pc.positions[3 * i + 1], pc.positions[3 * i + 2]];
      const X = [0, 1, 2].map((r) => cmp.s * (cmp.R[3 * r] * x[0] + cmp.R[3 * r + 1] * x[1] + cmp.R[3 * r + 2] * x[2]) + cmp.t[r]);
      const R = pose0.R,
        t = pose0.t;
      const z = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
      if (!(z > 0)) continue;
      const u = Math.round((gtCam.f * (R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0])) / z + gtCam.cx);
      const v = Math.round((gtCam.f * (R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1])) / z + gtCam.cy);
      if (u < 0 || v < 0 || u >= gtCam.width || v >= gtCam.height) continue;
      const zg = gt.depth[v * gtCam.width + u];
      if (!(zg > 0) || z > zg * 1.05) continue; // occluded in frame 0
      n++;
      if (Math.abs(z - zg) < 0.03 * zg) good++;
    }
    expect(n).toBeGreaterThan(200);
    expect(good / n).toBeGreaterThan(0.85);
    const mesh = await engine.mesh({ targetTriangles: 4000 });
    expect(mesh.triangleCount).toBeGreaterThan(500);
    expect(mesh.triangleCount).toBeLessThanOrEqual(4000);
  }, 300_000);

  it('exports PLY points / mesh, GLB and a COLMAP zip with images', async () => {
    const ply = decodePLY(await (await engine.exportModel('ply-points')).arrayBuffer());
    expect(ply.points!.count).toBe(engine.denseCloudResult!.count);
    const plyMesh = decodePLY(await (await engine.exportModel('ply-mesh')).arrayBuffer());
    expect(plyMesh.mesh!.triangleCount).toBe(engine.meshResult!.triangleCount);
    const glb = parseGLB(new Uint8Array(await (await engine.exportModel('glb')).arrayBuffer()));
    expect(glb.json).toBeTruthy();
    const zip = readZip(new Uint8Array(await (await engine.exportModel('colmap-zip')).arrayBuffer()));
    const names = zip.map((e) => e.name);
    expect(names).toEqual(expect.arrayContaining(['sparse/0/cameras.txt', 'sparse/0/images.txt', 'sparse/0/points3D.txt']));
    expect(names.filter((n) => n.startsWith('images/')).length).toBe(res.cameras.length);
    const imagesTxt = new TextDecoder().decode(zip.find((e) => e.name === 'sparse/0/images.txt')!.data);
    expect(imagesTxt.split('\n').filter((l) => /^\d/.test(l)).length).toBe(2 * res.cameras.length);
  }, 120_000);

  it('loadSession restores the stored models in a new engine', async () => {
    const e2 = new PhotogrammetryEngine({ mode: 'auto', quality: 'balanced' }, testConfig({ store }));
    const sum = await e2.loadSession('orbit');
    expect(sum.status).toBe('meshed');
    expect(sum.mode).toBe('object');
    expect(sum.quality).toBe('fast');
    expect(sum.sparse!.cameras.length).toBe(res.cameras.length);
    expect((sum.sparse!.cameras[0] as { frameIndex?: number }).frameIndex).toBe(res.cameras[0].frameIndex);
    expect(sum.dense!.count).toBe(engine.denseCloudResult!.count);
    expect(sum.mesh!.triangleCount).toBe(engine.meshResult!.triangleCount);
    const b = await e2.exportModel('colmap-zip');
    expect(b.size).toBeGreaterThan(1000);
    e2.dispose();
    engine.dispose();
  }, 120_000);
});

// ---------------------------------------------------------------------------------------------

describe('engine: resume, background mapping, visibility, cancel', () => {
  it('resumes a capture after a worker respawn (shared store) and refines everything', async () => {
    const cap = makeCapture({ frames: 44, arcDeg: 55 });
    const store = new MemoryEngineStore('resume');
    const opts = { mode: 'object' as const, quality: 'fast' as const, sessionId: 'resume', hfovPriorDeg: 60 };
    // first worker: background mapping (default) — model updates come as live events
    const a = new PhotogrammetryEngine(opts, { pipeline, store });
    const live: LiveUpdate[] = [];
    a.on('live', (u) => live.push(u));
    await a.init();
    for (const f of frameInputs(cap, { to: 24 })) {
      const u = await a.processFrame(f);
      // back-pressure: a keyframe frame returns only once ≤ maxMappingBacklog (default 2) keyframes are pending
      if (u.decision === 'keyframe') expect(a.liveSession!.pendingMapping).toBeLessThanOrEqual(2);
    }
    await a.flush();
    expect(live.some((u) => u.sparse && u.cameras)).toBe(true);
    const kfA = a.liveSession!.state.order.slice();
    expect(kfA.length).toBeGreaterThan(5);
    a.dispose();
    // respawned worker: resume
    const b = new PhotogrammetryEngine({ ...opts, resume: true }, testConfig({ store }));
    await b.init();
    const st = b.liveSession!.state;
    expect(st.order).toEqual(kfA);
    expect(st.imu.samples.length).toBeGreaterThan(100);
    expect(b.liveSession!.mapper.reconstruction.numRegisteredImages).toBeGreaterThanOrEqual(kfA.length - 1);
    // hidden page: frames are skipped (no GPU work)
    b.setHidden(true);
    const skipped = await b.processFrame(frameInputs(cap, { from: 24, to: 25 })[0]);
    expect(skipped.decision).toBe('skip');
    b.setHidden(false);
    for (const f of frameInputs(cap, { from: 25 })) await b.processFrame(f);
    await b.finishCapture();
    const ids = st.order;
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids.length).toBeGreaterThan(kfA.length);
    // cancel
    await expect(b.refine(undefined, { cancelled: true })).rejects.toMatchObject({ name: 'CancelledError' });
    const res = await b.refine();
    expect(res.cameras.length).toBeGreaterThanOrEqual(ids.length - 1);
    b.dispose();
  }, 300_000);
});

describe('engine: R_cam_device verification', () => {
  it('switches a wrong camera←device mapping using the gyro (frames reported at θ=180)', async () => {
    const cap = makeCapture({ frames: 40, arcDeg: 50 });
    const e = new PhotogrammetryEngine({ mode: 'object', quality: 'fast', hfovPriorDeg: 60 }, testConfig());
    const logs: string[] = [];
    e.on('log', (l, m) => logs.push(`${l}: ${m}`));
    await e.init();
    for (const f of frameInputs(cap, { screenAngle: 180 })) await e.processFrame(f);
    expect(e.liveSession!.state.thetaSession).toBe(180);
    expect(e.liveSession!.state.cameraTheta).toBe(0);
    expect(logs.some((l) => l.startsWith('warn: R_cam_device check'))).toBe(true);
    await e.finishCapture();
    const res = await e.refine();
    expect(res.metric.gravityAligned).toBe(true);
    expect(compare(cap, res).gravityDeg).toBeLessThan(1);
    e.dispose();
  }, 300_000);

  it('treats landscape frames in a portrait UI as raw sensor buffers (θ=90)', async () => {
    const cap = makeCapture({ frames: 40, arcDeg: 50, width: 320, height: 240, theta: 90 });
    const e = new PhotogrammetryEngine({ mode: 'object', quality: 'fast', hfovPriorDeg: 60 }, testConfig());
    await e.init();
    for (const f of frameInputs(cap, { screenAngle: 0 })) await e.processFrame(f);
    expect(e.liveSession!.state.thetaSession).toBe(90);
    expect(e.liveSession!.state.cameraTheta).toBe(90);
    expect(e.liveSession!.state.cameraThetaVerified).toBe(true);
    e.dispose();
  }, 300_000);

  it('handEyeCheck ranks the hypotheses from GT poses + IMU', () => {
    const cap = makeCapture({ frames: 40, render: false });
    const log = new ImuLog();
    log.load(cap.imu.slice());
    const kfs = [];
    for (let i = 0; i < 12; i++) {
      const t = cap.opts.t0 + i * 250;
      kfs.push({ t, pose: cap.poseAt(t) });
    }
    const r = handEyeCheck(kfs, log)!;
    expect(r).not.toBeNull();
    expect(r.best).toBe(0);
    expect(r.medianErrDeg[0]).toBeLessThan(0.05);
    expect(Math.min(r.medianErrDeg[1], r.medianErrDeg[2], r.medianErrDeg[3])).toBeGreaterThan(1);
  });
});

describe('capture mode classification (auto)', () => {
  it('orbits converge on an object, walks look outward', () => {
    const cap = makeCapture({ frames: 40, render: false });
    const orbit = cap.frames.filter((_, i) => i % 4 === 0).map((f) => f.pose);
    expect(classifyScene(orbit, 1.4)).toBe('object');
    // sideways walk along x, looking at +y (a facade 10 m away) with mild panning
    const walk: Pose[] = [];
    for (let i = 0; i < 10; i++) {
      const R = Float64Array.of(1, 0, 0, 0, 0, -1, 0, 1, 0); // x right, y down (−z world), z forward (+y world)
      const C = [i * 0.5, 0, 1.5];
      walk.push({ R, t: Float64Array.of(-(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]), -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]), -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2])) });
    }
    expect(classifyScene(walk, 10)).toBe('scene');
    expect(classifyScene(walk.slice(0, 3), 10)).toBeNull();
  });
});

describe('persistence records', () => {
  it('SparseModel packs and unpacks', () => {
    const m = {
      cameras: new Map([[0, { width: 10, height: 8, f: 9, cx: 4.5, cy: 3.5, k1: 0.01, k2: -0.002 }]]),
      images: [{ id: 3, cameraId: 0, pose: { R: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), t: Float64Array.of(1, 2, 3) } }],
      numPoints: 2,
      xyz: Float64Array.of(1, 2, 3, 4, 5, 6),
      colors: Uint8Array.of(1, 2, 3, 4, 5, 6),
      errors: Float32Array.of(0.5, 0.25),
      trackOffsets: Uint32Array.of(0, 2, 4),
      trackImages: Uint32Array.of(3, 4, 3, 5),
      trackFeatures: Uint32Array.of(0, 1, 2, 3),
    };
    const b = unpackSparseModel(packSparseModel(m));
    expect(b.cameras.get(0)).toEqual(m.cameras.get(0));
    expect(Array.from(b.images[0].pose.t)).toEqual([1, 2, 3]);
    expect(Array.from(b.xyz)).toEqual(Array.from(m.xyz));
    expect(Array.from(b.trackFeatures)).toEqual([0, 1, 2, 3]);
    expect(Array.from(b.errors)).toEqual([0.5, 0.25]);
  });
});
