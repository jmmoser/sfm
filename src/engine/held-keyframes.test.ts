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

  it('finishCapture keeps a frame still held when the capture ends (it is mapped like any keyframe)', async () => {
    const store = new MemoryEngineStore('held-end');
    const engine = new PhotogrammetryEngine(
      { mode: 'object', quality: 'fast', persist: false, sessionId: 'held-end', hfovPriorDeg: 60 },
      { pipeline, mapping: 'background', store, keyframe: { deferSharpnessRatio: 10, maxDeferMs: 150, hardOverlap: 0 } },
    );
    await engine.init();
    const live = () => engine.liveSession!;
    let heldIndex = -1;
    const frames = frameInputs(cap);
    for (let i = 0; i < frames.length; i++) {
      await engine.processFrame(frames[i]);
      // stop the capture right after a frame was held, once the live model is going
      if (live().selector.heldTime === cap.frames[i].t && live().state.order.length >= 6) {
        heldIndex = i;
        break;
      }
    }
    expect(heldIndex).toBeGreaterThan(0);
    const nBefore = live().state.order.length;
    const heldBefore = live().stats.heldKeyframes;
    await engine.finishCapture();
    const st = live().state;
    expect(st.order.length).toBe(nBefore + 1);
    const rec = st.records.get(st.order[st.order.length - 1])!;
    expect(rec.frameIndex).toBe(heldIndex);
    expect(rec.t).toBe(cap.frames[heldIndex].t);
    expect(live().stats.heldKeyframes).toBe(heldBefore + 1);
    expect(live().selector.heldTime).toBeNull();
    expect(live().pendingMapping).toBe(0);
    // mapped by the flush of finishCapture: registered in the live model
    expect(live().mapper.reconstruction.images.get(rec.id)?.registered).toBe(true);
    // the keyframe's image and record are stored
    await engine.flush();
    const stored = await store.getFrameRecords();
    expect(stored.some((r) => r.id === rec.id && r.frameIndex === heldIndex)).toBe(true);
    engine.dispose();
  }, 300_000);
});
