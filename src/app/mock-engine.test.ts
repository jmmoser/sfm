import { describe, expect, it } from 'vitest';
import { decodePLY, parseGLB, readZip } from '../io';
import type { FrameInput, LiveUpdate } from '../workers/protocol';
import { MockEngine } from './mock-engine';
import { MockScene, lookAt, poseCenter } from './mock-scene';

function fakeBitmap(w = 1440, h = 1920): { width: number; height: number; closed: boolean; close: () => void } {
  const b = { width: w, height: h, closed: false, close: () => (b.closed = true) };
  return b;
}

function frame(t: number, bm = fakeBitmap()): FrameInput {
  return { bitmap: bm as unknown as ImageBitmap, t, screenAngle: 0, imu: [] };
}

describe('mock scene', () => {
  it('lookAt gives an OpenCV world-to-camera pose looking at the target', () => {
    const p = lookAt([1, -2, 0.5], [0, 0, 0.2]);
    const R = p.R;
    // Orthonormal, det +1.
    const det = R[0] * (R[4] * R[8] - R[5] * R[7]) - R[1] * (R[3] * R[8] - R[5] * R[6]) + R[2] * (R[3] * R[7] - R[4] * R[6]);
    expect(det).toBeCloseTo(1, 12);
    expect(poseCenter(p).map((v) => +v.toFixed(12))).toEqual([1, -2, 0.5]);
    // The target projects onto the optical axis (x = y = 0, z > 0).
    const T = [0, 0, 0.2];
    const xc = R[0] * T[0] + R[1] * T[1] + R[2] * T[2] + p.t[0];
    const yc = R[3] * T[0] + R[4] * T[1] + R[5] * T[2] + p.t[1];
    const zc = R[6] * T[0] + R[7] * T[1] + R[8] * T[2] + p.t[2];
    expect(Math.abs(xc)).toBeLessThan(1e-12);
    expect(Math.abs(yc)).toBeLessThan(1e-12);
    expect(zc).toBeGreaterThan(0);
    // Camera +y (row 1 of R) points down: its world z component is negative.
    expect(R[5]).toBeLessThan(0);
  });

  it('samples are deterministic and the mesh is consistent', () => {
    const s = new MockScene('object', 3);
    const a = s.sample(500);
    const b = s.sample(500);
    expect(Array.from(a.positions)).toEqual(Array.from(b.positions));
    const m = s.mesh();
    expect(m.positions.length).toBe(3 * m.vertexCount);
    expect(m.indices.length).toBe(3 * m.triangleCount);
    expect(Math.max(...m.indices)).toBeLessThan(m.vertexCount);
    const d = s.dense(2000);
    expect(d.radii!.every((r) => r > 0 && r < 0.05)).toBe(true);
  });

  it('projected scene points land inside the frame', () => {
    const s = new MockScene('scene', 1);
    const smp = s.sample(4000);
    const pts = s.project(smp.positions, smp.count, s.poseAt(0.4), s.intrinsics(1920, 1440));
    expect(pts.length).toBeGreaterThan(40);
    for (const v of pts) expect(v >= 0 && v <= 1).toBe(true);
  });
});

describe('MockEngine', () => {
  it('runs capture → refine → densify → mesh → export with real data', async () => {
    const e = new MockEngine({ speed: 40, sessions: async () => null });
    const lives: LiveUpdate[] = [];
    e.onLive((u) => lives.push(u));
    const init = await e.init({ mode: 'object', quality: 'fast' });
    expect(init.adapter).toMatch(/Mock/);
    let t = 1000;
    let last: LiveUpdate | null = null;
    const bitmaps: ReturnType<typeof fakeBitmap>[] = [];
    for (let i = 0; i < 60 && (last?.numKeyframes ?? 0) < 8; i++) {
      t += 100;
      const bm = fakeBitmap();
      bitmaps.push(bm);
      last = await e.processFrame(frame(t, bm));
    }
    expect(bitmaps.every((b) => b.closed)).toBe(true); // the engine owns (and closes) frames
    expect(last!.numKeyframes).toBeGreaterThanOrEqual(8);
    expect(last!.tracking).toBe('tracking');
    expect(lives.length).toBe(bitmaps.length);
    const withSparse = lives.filter((u) => u.sparse);
    expect(withSparse.length).toBeGreaterThan(3);
    const sp = withSparse[withSparse.length - 1].sparse!;
    expect(sp.positions.length / 3).toBe(sp.colors.length / 3);
    // Overlay keypoints are normalised.
    const ov = lives.find((u) => u.overlay && u.overlay.length > 10)!.overlay!;
    for (const v of ov) expect(v >= 0 && v <= 1).toBe(true);

    await e.finishCapture();
    const stages: string[] = [];
    const sparse = await e.refine((p) => stages.push(p.stage));
    expect(new Set(stages)).toEqual(new Set(['matching', 'reconstruction', 'alignment']));
    expect(sparse.points.count).toBeGreaterThan(100);
    expect(sparse.cameras.length).toBeGreaterThanOrEqual(8);
    expect(sparse.metric.metric).toBe(true);
    const dense = await e.densify();
    expect(dense.count).toBeGreaterThan(10_000);
    expect(dense.normals).toBeDefined();
    const mesh = await e.mesh();
    expect(mesh.triangleCount).toBeGreaterThan(1000);

    const ply = decodePLY(await (await e.exportModel('ply-points')).arrayBuffer());
    expect(ply.points?.count).toBe(dense.count);
    const plyMesh = decodePLY(await (await e.exportModel('ply-mesh')).arrayBuffer());
    expect(plyMesh.mesh?.triangleCount).toBe(mesh.triangleCount);
    const glb = parseGLB(await (await e.exportModel('glb')).arrayBuffer());
    expect(glb.json).toBeTruthy();
    const zip = readZip(new Uint8Array(await (await e.exportModel('colmap-zip')).arrayBuffer()));
    const names = zip.map((z) => z.name);
    expect(names).toContain('sparse/0/cameras.txt');
    expect(names).toContain('sparse/0/images.txt');
    expect(names).toContain('sparse/0/points3D.txt');
    // the USDZ atlas is JPEG-encoded with OffscreenCanvas (browser: src/app/ar-quicklook.gpu.test.ts)
    await expect(e.exportModel('usdz')).rejects.toThrow(/OffscreenCanvas/);
  });

  it('back-pressure: a concurrent frame is rejected and its bitmap closed', async () => {
    const e = new MockEngine({ speed: 1, sessions: async () => null });
    await e.init({ mode: 'scene', quality: 'fast' });
    const p1 = e.processFrame(frame(100));
    expect(e.frameInFlight).toBe(true);
    const bm = fakeBitmap();
    await expect(e.processFrame(frame(200, bm))).rejects.toMatchObject({ name: 'EngineBusyError' });
    expect(bm.closed).toBe(true);
    await p1;
    expect(e.frameInFlight).toBe(false);
  });

  it('cancel rejects a long operation with CancelledError', async () => {
    const e = new MockEngine({ speed: 1, sessions: async () => null });
    await e.init({ mode: 'object', quality: 'balanced' });
    const p = e.refine();
    setTimeout(() => e.cancel(), 150);
    await expect(p).rejects.toMatchObject({ name: 'CancelledError' });
  });

  it('finishCapture is a cancellable long operation; a stalled one is aborted with a simulated restart', async () => {
    const e = new MockEngine({ speed: 1, sessions: async () => null });
    await e.init({ mode: 'object', quality: 'fast' });
    const p = e.finishCapture();
    setTimeout(() => e.cancel(), 50);
    await expect(p).rejects.toMatchObject({ name: 'CancelledError' });
    expect(await e.abortLongOperation()).toBe('none');

    const stuck = new MockEngine({ speed: 20, stallOn: { finishCapture: true }, sessions: async () => null });
    await stuck.init({ mode: 'object', quality: 'fast' });
    const states: string[] = [];
    stuck.onStatus((s) => states.push(`${s.state}:${s.reason ?? ''}`));
    const q = stuck.finishCapture().catch((err: Error) => err);
    await new Promise((r) => setTimeout(r, 30));
    expect(await stuck.abortLongOperation({ graceMs: 100 })).toBe('restarted');
    expect(await q).toMatchObject({ name: 'CancelledError' });
    expect(states[0]).toMatch(/^respawning:aborted: /);
    expect(stuck.status.state).toBe('ready');
    // The session survives the "restart": processing works.
    expect((await stuck.refine()).cameras.length).toBeGreaterThan(0);
  });

  it('abort with reinit: false restarts without re-opening the session; release() forgets it', async () => {
    const stuck = new MockEngine({ speed: 20, stallOn: { finishCapture: true }, sessions: async () => null });
    await stuck.init({ mode: 'object', quality: 'fast', sessionId: 'gone' });
    const q = stuck.finishCapture().catch((err: Error) => err);
    await new Promise((r) => setTimeout(r, 30));
    expect(await stuck.abortLongOperation({ graceMs: 100, reinit: false })).toBe('restarted');
    expect(await q).toMatchObject({ name: 'CancelledError' });
    expect(stuck.status.state).toBe('idle');
    await expect(stuck.refine()).rejects.toMatchObject({ name: 'EngineNotInitializedError' });

    const e = new MockEngine({ speed: 50, sessions: async () => null });
    await e.init({ mode: 'object', quality: 'fast' });
    for (let i = 0; i < 20; i++) await e.processFrame(frame(100 + i * 400));
    const r = e.refine().catch((err: Error) => err);
    await e.release();
    expect(await r).toMatchObject({ name: 'CancelledError' });
    expect(e.status).toMatchObject({ state: 'idle', reason: 'released' });
    await expect(e.refine()).rejects.toMatchObject({ name: 'EngineNotInitializedError' });
    // a new init works as before
    await e.init({ mode: 'object', quality: 'fast' });
    expect(e.status.state).toBe('ready');
  });

  it('closeSession lets go of that session only (like the worker: a running operation ends)', async () => {
    const e = new MockEngine({ speed: 50, sessions: async () => null });
    await e.init({ mode: 'object', quality: 'fast', sessionId: 's1' });
    for (let i = 0; i < 20; i++) await e.processFrame(frame(100 + i * 400));
    await e.closeSession('other');
    expect(e.status.state).toBe('ready');
    const r = e.refine().catch((err: Error) => err);
    await e.closeSession('s1');
    expect(await r).toMatchObject({ name: 'CancelledError' });
    expect(e.status).toMatchObject({ state: 'idle', reason: 'closed' });
    await expect(e.refine()).rejects.toMatchObject({ name: 'EngineNotInitializedError' });
  });

  it('closeSession during the init or load of that scan ends it, like the worker (its engine is disposed)', async () => {
    const meta = { id: 's1', name: 'Vase', createdAt: 1, updatedAt: 2, mode: 'object' as const, status: 'dense' as const, numFrames: 20 };
    const e = new MockEngine({ speed: 4, sessions: async () => ({ getSession: async () => meta }) as never });
    const init = e.init({ mode: 'object', quality: 'fast', sessionId: 'A' }).catch((err: Error) => err);
    await e.closeSession('A');
    expect(await init).toMatchObject({ name: 'EngineCrashedError', message: 'Engine disposed' });
    expect(e.status).toMatchObject({ state: 'idle', reason: 'closed' });
    const load = e.loadSession('s1').catch((err: Error) => err);
    await e.closeSession('s1');
    expect(await load).toMatchObject({ name: 'EngineCrashedError' });
    expect(e.status).toMatchObject({ state: 'idle', reason: 'closed' });
    // a close of another scan does not end an init
    const b = e.init({ mode: 'object', quality: 'fast', sessionId: 'B' });
    await e.closeSession('A');
    await b;
    expect(e.status.state).toBe('ready');
  });

  it('memoryLimitKeyframes: one out-of-memory event, then tracking only (no new keyframes)', async () => {
    const e = new MockEngine({ speed: 60, memoryLimitKeyframes: 3, sessions: async () => null });
    await e.init({ mode: 'object', quality: 'fast' });
    const errors: { reason?: string; name: string; fatal: boolean }[] = [];
    e.onError((x) => errors.push({ reason: x.reason, name: x.error.name, fatal: x.fatal }));
    let u: LiveUpdate | null = null;
    for (let i = 0; i < 120; i++) u = await e.processFrame(frame(100 + i * 100));
    expect(u!.numKeyframes).toBe(3);
    expect(u!.timings.memoryLimited).toBe(1);
    expect(u!.tracking).toBe('tracking');
    expect(errors).toEqual([{ reason: 'out-of-memory', name: 'CaptureMemoryLimitError', fatal: false }]);
  });

  it('an interrupted densify is reported as resumable (stats.denseResumablePass) until it completes', async () => {
    const e = new MockEngine({ speed: 4, sessions: async () => null });
    await e.init({ mode: 'object', quality: 'fast', sessionId: 'r1' });
    for (let i = 0; i < 40; i++) await e.processFrame(frame(100 + i * 400));
    await e.refine();
    const d = e.densify();
    setTimeout(() => e.cancel(), 100);
    await expect(d).rejects.toMatchObject({ name: 'CancelledError' });
    expect((await e.loadSession('r1')).stats).toMatchObject({ denseResumablePass: 1, denseResumableQuality: 'fast' });
    const pc = await e.densify();
    expect(pc.count).toBeGreaterThan(0);
    const sum = await e.loadSession('r1');
    expect(sum.stats.denseResumablePass).toBeUndefined();
    expect(sum.stats.denseResumedPass).toBe(1);
    expect(typeof sum.stats.memoryMB).toBe('number');
    expect(sum.stats.yawSource).toBe('compass');
  });

  it('splat: prepare → training (detail) → finishing; a cancel is resumable; model and ply-splat export', async () => {
    const e = new MockEngine({ speed: 2, sessions: async () => null });
    await e.init({ mode: 'object', quality: 'fast', sessionId: 's1' });
    for (let i = 0; i < 40; i++) await e.processFrame(frame(100 + i * 400));
    await e.refine();
    await expect(e.splatModel()).resolves.toBeNull();
    let cancelAt = 0;
    const s = e.splat(undefined, (p) => {
      if (p.stage === 'splat_training' && p.detail && p.detail.iteration > 500 && !cancelAt) {
        cancelAt = p.detail.iteration;
        e.cancel();
      }
    });
    await expect(s).rejects.toMatchObject({ name: 'CancelledError' });
    expect((await e.loadSession('s1')).stats).toMatchObject({ splatResumableIteration: cancelAt, splatResumableQuality: 'fast' });
    const stages: string[] = [];
    const details: Record<string, number>[] = [];
    const info = await e.splat(undefined, (p) => {
      if (!stages.includes(p.stage)) stages.push(p.stage);
      if (p.detail) details.push(p.detail);
    });
    expect(stages).toEqual(['splat_prepare', 'splat_training', 'splat_finishing']);
    expect(details[0].iteration).toBeGreaterThanOrEqual(cancelAt);
    expect(details[details.length - 1]).toMatchObject({ iteration: 3000, iterations: 3000 });
    expect(info).toMatchObject({ iterations: 3000, shDegree: 0, antialiased: true });
    expect(info.stats.splatResumedFrom).toBe(cancelAt);
    const payload = await e.splatModel();
    expect(payload?.model.count).toBe(info.count);
    expect((await e.loadSession('s1')).splat?.count).toBe(info.count);
    const ply = new TextDecoder().decode(new Uint8Array(await (await e.exportModel('ply-splat')).arrayBuffer()).subarray(0, 400));
    expect(ply).toContain(`element vertex ${info.count}`);
    expect(ply).toContain('property float f_dc_0');
    await e.refine(); // new poses: the splats are gone
    await expect(e.splatModel()).resolves.toBeNull();
  });

  it('failOn makes a method fail (error UI)', async () => {
    const e = new MockEngine({ speed: 50, failOn: { refine: 'out of memory' }, sessions: async () => null });
    await e.init({ mode: 'object', quality: 'fast' });
    await expect(e.refine()).rejects.toThrow('out of memory');
  });

  it('loadSession rebuilds stored models from the session status', async () => {
    const meta = { id: 's1', name: 'Vase', createdAt: 1, updatedAt: 2, mode: 'object' as const, status: 'dense' as const, numFrames: 20 };
    const e = new MockEngine({ speed: 50, sessions: async () => ({ getSession: async (id: string) => (id === 's1' ? meta : null) }) as never });
    const sum = await e.loadSession('s1');
    expect(sum.name).toBe('Vase');
    expect(sum.status).toBe('dense');
    expect(sum.sparse).not.toBeNull();
    expect(sum.dense).not.toBeNull();
    expect(sum.mesh).toBeNull();
    expect(sum.splat).toBeNull();
    await expect(e.loadSession('nope')).rejects.toThrow(/not found/);
  });

  it('loadSession restores the splats recorded in the session stats (and a resumable run)', async () => {
    const meta = { id: 's2', name: 'Vase', createdAt: 1, updatedAt: 2, mode: 'object' as const, status: 'dense' as const, numFrames: 20, stats: { splatGaussians: 5000, splatIterations: 5000, splatResumableIteration: 700, splatResumableQuality: 'max' } };
    const e = new MockEngine({ speed: 50, sessions: async () => ({ getSession: async (id: string) => (id === 's2' ? meta : null) }) as never });
    const sum = await e.loadSession('s2');
    expect(sum.splat).toMatchObject({ iterations: 5000, antialiased: true });
    expect(sum.stats).toMatchObject({ splatResumableIteration: 700, splatResumableQuality: 'max' });
    const payload = await e.splatModel();
    expect(payload!.model.count).toBe(sum.splat!.count);
    // another quality starts over (the resume needs the interrupted run's settings)
    const info = await e.splat({ quality: 'fast', iterations: 200 });
    expect(info.stats.splatResumedFrom).toBeUndefined();
  });

  it('injectSession serves real models as the in-memory session (screenshot harness)', async () => {
    const e = new MockEngine({ speed: 50, sessions: async () => null });
    const cloud = { count: 2, positions: new Float32Array([0, 0, 0, 1, 1, 1]), colors: new Uint8Array(6) };
    const cam = { width: 64, height: 48, f: 50, cx: 31.5, cy: 23.5, k1: 0, k2: 0 };
    const sparse = { points: cloud, cameras: [{ id: 0, pose: { R: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), t: new Float64Array(3) }, intrinsics: cam }], metric: { metric: true, gravityAligned: true, northAligned: false, source: 'imu', relativeUncertainty: 0.02, originLatLonAlt: null }, stats: {} };
    const m = { count: 1, shDegree: 0 as const, means: new Float32Array(3), logScales: new Float32Array(3), quats: new Float32Array([1, 0, 0, 0]), opacities: new Float32Array(1), sh: new Float32Array(3) };
    e.injectSession('real', { name: 'Real run', mode: 'object', sparse: sparse as never, dense: cloud, mesh: null, splat: m });
    const sum = await e.loadSession('real');
    expect(sum.name).toBe('Real run');
    expect(sum.dense).toBe(cloud);
    expect(sum.splat?.count).toBe(1);
    expect((await e.splatModel())!.model.count).toBe(1);
  });
});
