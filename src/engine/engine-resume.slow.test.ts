/**
 * PhotogrammetryEngine end to end on the CPU path (Node): resume after a worker respawn,
 * background mapping, visibility and cancellation, and the R_cam_device hand-eye check
 * switching a wrong camera←device mapping during capture.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import type { FeaturePipeline } from '../features/extractor';
import type { LiveUpdate } from '../workers/protocol';
import { createCpuFeaturePipeline } from './cpu-pipeline';
import { PhotogrammetryEngine } from './engine';
import { MemoryEngineStore } from './persistence';
import { compareWithGT as compare, frameInputs, makeCapture } from './testutil';
import type { EngineConfig } from './types';

let pipeline: FeaturePipeline;
beforeAll(() => {
  pipeline = createCpuFeaturePipeline({ maxFeatures: 2048, maxImageSize: 1024 });
});

function testConfig(extra: EngineConfig = {}): EngineConfig {
  return { pipeline, mapping: 'inline', ...extra };
}

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
});
