/**
 * Best-of-window keyframing in the engine (LiveSession + KeyframeSelector `hold` / `useHeld`): a
 * keyframe made from an earlier, HELD frame must carry that frame's identity everywhere — its
 * record (frameIndex, t → GT pose / IMU time), the update's `timings.keyframeFrameIndex` (harness
 * live-trajectory mapping) — and keyframes stay in time order. Every soft trigger is deferred
 * here (deferSharpnessRatio 10, hardOverlap 0), so the window's best frame is used whenever it beats the frame at
 * the deadline.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import type { FeaturePipeline } from '../features/extractor';
import type { LiveUpdate } from '../workers/protocol';
import { createCpuFeaturePipeline } from './cpu-pipeline';
import { PhotogrammetryEngine } from './engine';
import { MemoryEngineStore } from './persistence';
import { frameInputs, makeCapture, type SynthCapture } from './testutil';

let pipeline: FeaturePipeline;
let cap: SynthCapture;
beforeAll(() => {
  pipeline = createCpuFeaturePipeline({ maxFeatures: 2048, maxImageSize: 1024 });
  cap = makeCapture({ frames: 48 });
}, 120_000);

describe('best-of-window keyframes in the engine', () => {
  it('a keyframe made from a held frame has that frame’s index / time, and the update names it', async () => {
    const engine = new PhotogrammetryEngine(
      { mode: 'object', quality: 'fast', persist: false, sessionId: 'held', hfovPriorDeg: 60 },
      { pipeline, mapping: 'inline', store: new MemoryEngineStore('held'), keyframe: { deferSharpnessRatio: 10, maxDeferMs: 150, hardOverlap: 0 } },
    );
    await engine.init();
    const live = () => engine.liveSession!;
    const updates: LiveUpdate[] = [];
    const created: { update: LiveUpdate; record: { frameIndex: number; t: number } }[] = [];
    for (const f of frameInputs(cap)) {
      const before = live() ? live().state.order.length : 0;
      const u = await engine.processFrame(f);
      updates.push(u);
      const st = live().state;
      if (st.order.length > before) {
        expect(u.decision).toBe('keyframe');
        const r = st.records.get(st.order[st.order.length - 1])!;
        created.push({ update: u, record: { frameIndex: r.frameIndex, t: r.t } });
      }
    }
    const held = created.filter((c) => c.update.timings.keyframeFrameIndex !== undefined);
    expect(live().stats.heldKeyframes).toBe(held.length);
    expect(held.length).toBeGreaterThan(0);
    for (const c of created) {
      const k = c.update.timings.keyframeFrameIndex;
      if (k === undefined) {
        expect(c.record.frameIndex).toBe(c.update.frameIndex);
      } else {
        expect(k).toBeLessThan(c.update.frameIndex);
        expect(c.record.frameIndex).toBe(k);
        expect(c.record.t).toBe(cap.frames[k].t);
      }
    }
    // keyframes in strictly increasing time, never the same frame twice
    const ts = live().state.order.map((id) => live().state.records.get(id)!.t);
    for (let i = 1; i < ts.length; i++) expect(ts[i]).toBeGreaterThan(ts[i - 1]);
    // the live model is built from the right frames (registration works through the held ones)
    expect(updates[updates.length - 1].numRegistered).toBeGreaterThanOrEqual(0.8 * live().state.order.length);
    engine.dispose();
  }, 300_000);
});
