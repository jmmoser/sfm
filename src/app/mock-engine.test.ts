import { describe, expect, it } from 'vitest';
import { decodePLY, parseGLB, readZip } from '../io';
import type { FrameInput, LiveUpdate, SplatResult } from '../workers/protocol';
import { computeCoverage } from './coverage';
import { decodeSplatSPZ } from '../splat/spz';
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
  it("auto resolves to 'object' after 8 keyframes; a level orbit, then the higher loop, fill the coverage ring", async () => {
    const e = new MockEngine({ speed: 8, sessions: async () => null });
    await e.init({ mode: 'auto', quality: 'fast' });
    let t = 1000;
    // the model comes in the standalone (background-mapping) updates
    const model: { afterLoop: LiveUpdate | null; last: LiveUpdate | null } = { afterLoop: null, last: null };
    e.onLive((u) => {
      if (!u.cameras) return;
      model.last = u;
      if (e.scanProgress < 1.05) model.afterLoop = u;
    });
    const modes: string[] = [];
    while (e.scanProgress < 2.3) {
      t += 100;
      const u = await e.processFrame(frame(t));
      modes.push(`${u.numKeyframes}:${u.resolvedMode}`);
    }
    expect(modes[0]).toBe('1:auto'); // the first frame is a keyframe
    expect(modes.find((m) => m.endsWith('object'))).toBe('8:object');
    expect(modes.filter((m) => m.endsWith('auto')).every((m) => Number(m.split(':')[0]) < 8)).toBe(true);
    // the mock cameras carry no upCam: up from the camera axes (the fallback path)
    const loop = computeCoverage(model.afterLoop!.cameras!);
    expect(loop.stable).toBe(true);
    expect(loop.covered[1]).toBeGreaterThanOrEqual(11);
    expect(loop.covered[2]).toBe(0);
    expect(loop.prompt).toBe('look_down');
    const both = computeCoverage(model.last!.cameras!);
    expect(both.covered[1]).toBe(12);
    expect(both.covered[2]).toBeGreaterThanOrEqual(10);
    expect(both.prompt).toBe('done');
    e.dispose();
  });

  it("sends a keyframe's model like background mapping: under the in-flight frame's index, before that frame's response", async () => {
    const e = new MockEngine({ speed: 8, sessions: async () => null });
    await e.init({ mode: 'object', quality: 'fast' });
    const lives: LiveUpdate[] = [];
    e.onLive((u) => lives.push(u));
    const responses: LiveUpdate[] = [];
    let t = 1000;
    while (responses.filter((r) => r.decision === 'keyframe').length < 6) {
      t += 100;
      responses.push(await e.processFrame(frame(t)));
    }
    expect(responses.every((r) => !r.cameras && !r.sparse)).toBe(true);
    const models = lives.filter((u) => !responses.includes(u));
    expect(models.length).toBe(4); // keyframes 2–5 (the 6th is pending)
    for (const m of models) {
      const i = lives.indexOf(m);
      const [prev, next] = [lives[i - 1], lives[i + 1]];
      // the keyframe's own response, then the model, then the response of the frame in flight
      expect(prev.decision).toBe('keyframe');
      expect(responses).toContain(next);
      expect(m.frameIndex).toBe(next.frameIndex);
      expect([m.decision, m.guidance]).toEqual([prev.decision, prev.guidance]);
      expect(m.cameras!.length).toBe(prev.numKeyframes);
      expect(m.sparse!.positions.length).toBeGreaterThan(0);
      expect(m.overlay).toBeUndefined();
      expect(m.framePose).toBeUndefined();
    }
    // finishCapture flushes the pending one
    await e.finishCapture();
    const flushed = lives[lives.length - 1];
    expect(responses).not.toContain(flushed);
    expect(flushed.cameras!.length).toBe(6);
    expect(flushed.frameIndex).toBe(responses[responses.length - 1].frameIndex);
    e.dispose();
  });

  it('runs capture → refine → densify → mesh → export with real data', async () => {
    const e = new MockEngine({ speed: 40, sessions: async () => null });
    const lives: LiveUpdate[] = [];
    e.onLive((u) => lives.push(u));
    const init = await e.init({ mode: 'object', quality: 'fast' });
    expect(init.adapter).toMatch(/Mock/);
    let t = 1000;
    let last: LiveUpdate | null = null;
    const bitmaps: ReturnType<typeof fakeBitmap>[] = [];
    const responses: LiveUpdate[] = [];
    for (let i = 0; i < 60 && (last?.numKeyframes ?? 0) < 8; i++) {
      t += 100;
      const bm = fakeBitmap();
      bitmaps.push(bm);
      last = await e.processFrame(frame(t, bm));
      responses.push(last);
    }
    expect(bitmaps.every((b) => b.closed)).toBe(true); // the engine owns (and closes) frames
    expect(last!.numKeyframes).toBeGreaterThanOrEqual(8);
    expect(last!.tracking).toBe('tracking');
    // every frame response is also a live event; the others are model updates
    expect(responses.every((r) => lives.includes(r))).toBe(true);
    expect(lives.filter((u) => !responses.includes(u)).every((u) => u.sparse && u.cameras)).toBe(true);
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

  it('splat: prepare → training (detail) → finishing; a cancel is resumable; model and ply-splat / spz exports', async () => {
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
    const spz = await decodeSplatSPZ(new Uint8Array(await (await e.exportModel('spz')).arrayBuffer()));
    expect(spz.model.count).toBe(info.count);
    await e.refine(); // new poses: the splats are gone
    await expect(e.splatModel()).resolves.toBeNull();
  });

  it('splat previews; finishEarly keeps that model and a later run continues from it', async () => {
    // (slow enough for a preview before iteration 900: one every 400 ms of a 1.6 s training)
    const e = new MockEngine({ speed: 1, sessions: async () => null });
    await e.init({ mode: 'object', quality: 'fast', sessionId: 's3' });
    for (let i = 0; i < 40; i++) await e.processFrame(frame(100 + i * 400));
    await e.refine();
    const previews: { iteration: number; sum: number; w: number; h: number }[] = [];
    let requested = 0;
    let firstTraining = -1;
    const info = await e.splat({ preview: true }, (p) => {
      if (p.stage === 'splat_preview' && p.image && p.detail) {
        previews.push({ iteration: p.detail.iteration, sum: Array.from(p.image.data).reduce((s, v) => s + v, 0), w: p.image.width, h: p.image.height });
      }
      if (p.stage === 'splat_training' && p.detail) {
        if (firstTraining < 0) firstTraining = previews.length;
        expect(p.detail.finishFrom).toBe(900);
        // a request before finishFrom waits for it
        if (!requested) {
          requested = p.detail.iteration;
          e.finishEarly('splat');
        }
      }
    });
    expect(requested).toBeLessThan(900);
    expect(firstTraining).toBe(1); // the first preview comes before the training
    expect(previews[0]).toMatchObject({ iteration: 0, w: 288, h: 384 });
    expect(previews.length).toBeGreaterThanOrEqual(2);
    expect(new Set(previews.map((p) => p.sum)).size).toBe(previews.length); // they change
    expect(info.iterations).toBeGreaterThanOrEqual(900);
    expect(info.iterations).toBeLessThan(3000);
    expect(info.stats).toMatchObject({ splatFinishedEarly: 1, splatContinueFrom: info.iterations, splatIterations: 3000 });
    const sum = await e.loadSession('s3');
    expect(sum.splat?.iterations).toBe(info.iterations);
    expect(sum.stats).toMatchObject({ splatResumableIteration: info.iterations, splatResumableQuality: 'fast', splatContinuesModel: 1 });
    // continue: from there to the end, no longer resumable
    const done = await e.splat();
    expect(done.iterations).toBe(3000);
    expect(done.stats.splatResumedFrom).toBe(info.iterations);
    expect(done.stats.splatFinishedEarly).toBeUndefined();
    const end = (await e.loadSession('s3')).stats;
    expect(end.splatResumableIteration).toBeUndefined();
    expect(end.splatContinuesModel).toBeUndefined();
  });

  it('a continued run cancelled later, or a new dense cloud, ends "continue" (review)', async () => {
    const e = new MockEngine({ speed: 4, sessions: async () => null });
    await e.init({ mode: 'object', quality: 'fast', sessionId: 's4' });
    for (let i = 0; i < 40; i++) await e.processFrame(frame(100 + i * 400));
    await e.refine();
    await e.densify();
    const endEarly = (): Promise<SplatResult> =>
      e.splat(undefined, (p) => {
        if (p.stage === 'splat_training') e.finishEarly('splat');
      });
    const early = await endEarly();
    expect((await e.loadSession('s4')).stats.splatContinuesModel).toBe(1);
    // continued, then cancelled further on: an interrupted run's checkpoint
    const cancelled = await e
      .splat(undefined, (p) => {
        if (p.stage === 'splat_training' && p.detail && p.detail.iteration > early.iterations + 100) e.cancel();
      })
      .catch((x: unknown) => x);
    expect((cancelled as Error).name).toBe('CancelledError');
    const st = (await e.loadSession('s4')).stats;
    expect(st.splatResumableIteration).toBeGreaterThan(early.iterations);
    expect(st.splatContinuesModel).toBeUndefined();
    // ended early again, then the dense cloud is rebuilt: its checkpoint is gone (a new run)
    await e.splat({ restart: true }, (p) => {
      if (p.stage === 'splat_training') e.finishEarly('splat');
    });
    expect((await e.loadSession('s4')).stats.splatContinuesModel).toBe(1);
    await e.densify({ quality: 'balanced' });
    const after = (await e.loadSession('s4')).stats;
    expect(after.splatResumableIteration).toBeUndefined();
    expect(after.splatContinuesModel).toBeUndefined();
    expect((await e.loadSession('s4')).splat).not.toBeNull(); // the model stays
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

  it('resume markers survive a reload in the session record (densify after a stored map, splat checkpoints) and go when the run ends', async () => {
    const rec = { id: 'k1', name: 'Wall', createdAt: 1, updatedAt: 2, mode: 'object' as const, status: 'sparse' as const, numFrames: 20, stats: { quality: 'fast' } as Record<string, number | string> };
    const store = {
      getSession: async (id: string) => (id === rec.id ? { ...rec, stats: { ...rec.stats } } : null),
      updateSession: async (_id: string, patch: { stats?: Record<string, number | string> }) => Object.assign(rec, patch),
    };
    const sessions = async () => store as never;
    // a densify killed after its first depth map (the page reloads: a new engine)
    const a = new MockEngine({ speed: 1, sessions });
    await a.loadSession('k1');
    let resolveTick!: () => void;
    const ticked = new Promise<void>((r) => (resolveTick = r));
    const dense = a.densify(undefined, (p) => {
      if (p.stage === 'depth_maps' && p.progress > 0.1) resolveTick();
    });
    await ticked;
    a.dispose();
    await dense.catch(() => undefined);
    expect(rec.stats).toMatchObject({ denseResumablePass: 1, denseResumableQuality: 'fast' });
    const b = new MockEngine({ speed: 20, sessions });
    expect((await b.loadSession('k1')).stats).toMatchObject({ denseResumablePass: 1, denseResumableQuality: 'fast' });
    await b.densify();
    expect(rec.stats.denseResumablePass).toBeUndefined();
    expect(rec.stats.denseResumableQuality).toBeUndefined();

    // a photoreal build killed after its first checkpoint resumes from it
    let resolveCkpt!: () => void;
    const pastCkpt = new Promise<void>((r) => (resolveCkpt = r));
    const c = new MockEngine({ speed: 1, sessions });
    await c.loadSession('k1');
    const train = c.splat({ quality: 'fast' }, (p) => {
      if (p.detail && p.detail.iteration > 700) resolveCkpt();
    });
    await pastCkpt;
    c.dispose();
    await train.catch(() => undefined);
    expect(rec.stats).toMatchObject({ splatResumableIteration: 500, splatResumableQuality: 'fast' });
    const d = new MockEngine({ speed: 20, sessions });
    expect((await d.loadSession('k1')).stats).toMatchObject({ splatResumableIteration: 500 });
    const info = await d.splat({ quality: 'fast' });
    expect(info.stats.splatResumedFrom).toBe(500);
    expect(rec.stats.splatResumableIteration).toBeUndefined();
    // refine removes every marker (new poses)
    rec.stats = { ...rec.stats, denseResumablePass: 1, splatResumableIteration: 900 };
    await d.refine();
    expect(Object.keys(rec.stats).filter((k) => /Resumable/.test(k))).toEqual([]);
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

describe('MockEngine: continue a stored scan (init with resume + extend)', () => {
  async function capture(e: MockEngine, t0: number, keyframes: number): Promise<{ t: number; last: LiveUpdate }> {
    let t = t0;
    let last: LiveUpdate | null = null;
    for (let i = 0; i < 400 && (last?.numKeyframes ?? 0) < keyframes; i++) {
      t += 100;
      last = await e.processFrame(frame(t));
    }
    return { t, last: last! };
  }

  it('continues the scan held in memory after its last keyframe; refine covers every keyframe', async () => {
    const e = new MockEngine({ speed: 40, sessions: async () => null, unconnectedImages: 2 });
    await e.init({ mode: 'object', quality: 'fast', sessionId: 'x', persist: true });
    const first = await capture(e, 1000, 8);
    await e.finishCapture();
    const s1 = await e.refine();
    expect(s1.stats).toMatchObject({ unconnectedImages: 2, registeredImages: (s1.stats.images as number) - 2 });
    const sAt = e.scanProgress;
    // a reloaded page: the clock restarts
    const init = await e.init({ mode: 'scene', quality: 'max', sessionId: 'x', persist: true, resume: true, extend: true });
    expect(init.resumedKeyframes).toBe(first.last.numKeyframes);
    expect(init.captureMemory!.capMB).toBeGreaterThan(init.captureMemory!.usedMB);
    expect(e.captureFinished).toBe(false);
    expect(e.scanProgress).toBeCloseTo(sAt, 1);
    const second = await capture(e, 500, first.last.numKeyframes + 4);
    expect(second.last.numKeyframes).toBe(first.last.numKeyframes + 4);
    const s2 = await e.refine();
    expect(s2.stats.images).toBeGreaterThanOrEqual(second.last.numKeyframes);
  });

  it('rebuilds a stored scan from its session record and rejects an unknown one', async () => {
    const meta = { id: 's3', name: 'Vase', createdAt: 1, updatedAt: 2, mode: 'object' as const, status: 'meshed' as const, numFrames: 0, stats: { keyframes: 14, quality: 'fast' } };
    const e = new MockEngine({ speed: 50, sessions: async () => ({ getSession: async (id: string) => (id === 's3' ? meta : null) }) as never });
    const init = await e.init({ mode: 'object', quality: 'balanced', sessionId: 's3', persist: true, resume: true, extend: true });
    expect(init.resumedKeyframes).toBe(14);
    // the stored model for the capture view (the engine's InitResult.storedModel)
    expect(init.storedModel!.cameras).toHaveLength(14);
    expect(init.storedModel!.sparse.positions.length).toBeGreaterThan(0);
    const { last } = await capture(e, 100, 16);
    expect(last.numKeyframes).toBe(16);
    await expect(e.init({ mode: 'object', quality: 'fast', sessionId: 'gone', persist: true, resume: true, extend: true })).rejects.toThrow(/no stored capture/);
  });
});
