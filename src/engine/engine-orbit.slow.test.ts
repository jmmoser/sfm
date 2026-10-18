/**
 * PhotogrammetryEngine end to end on the CPU path (Node), on a synthetic hand-held orbit
 * rendered with the dense ray caster + exact IMU (src/engine/testutil.ts): keyframing, live
 * mapping, refine (registration, self-calibration, world alignment with the simulated IMU:
 * gravity, metric scale, time offset), dense + mesh + exports and loadSession.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import type { Pose } from '../core/types';
import { pinholeIntrinsics, renderSynthetic } from '../dense/synthetic';
import type { FeaturePipeline } from '../features/extractor';
import { parseGLB } from '../io/glb';
import { decodePLY } from '../io/ply';
import { readZip } from '../io/zip';
import type { LiveUpdate } from '../workers/protocol';
import { createCpuFeaturePipeline } from './cpu-pipeline';
import { PhotogrammetryEngine } from './engine';
import { MemoryEngineStore } from './persistence';
import { compareWithGT as compare, frameInputs, makeCapture, objectScene, type SynthCapture } from './testutil';
import type { EngineConfig, SparseResultEx } from './types';

let pipeline: FeaturePipeline;
beforeAll(() => {
  pipeline = createCpuFeaturePipeline({ maxFeatures: 2048, maxImageSize: 1024 });
});

function testConfig(extra: EngineConfig = {}): EngineConfig {
  return { pipeline, mapping: 'inline', ...extra };
}

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
