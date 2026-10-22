/**
 * runPipeline (Node) against a scripted stub engine (./engines is mocked): the live summary built
 * from frame responses and unsolicited background model updates, its object coverage, and the
 * delivered count of an aborted capture. fetch serves meta.json / frames.json / the frames (other files 404),
 * createImageBitmap is stubbed, and refine throws so the run ends after finishCapture.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { lookAt } from '../../app/mock-scene';
import type { InitResult, LiveCamera, LiveUpdate } from '../../workers/protocol';
import type { HarnessEngine } from './engines';
import type { HarnessConfig } from './types';

const h = vi.hoisted(() => ({ engine: null as unknown }));

vi.mock('./engines', () => ({ createHarnessEngine: async () => h.engine }));

import { runPipeline } from './pipeline';

type Emit = (u: LiveUpdate) => void;

interface Script {
  /** Frame response to delivery k (FrameInput.index); `emit` sends unsolicited updates before it. */
  frame(k: number, emit: Emit): LiveUpdate;
  finish?(emit: Emit): void;
}

/** A HarnessEngine that forwards every frame response to onLive, like the real back-ends. */
function stubEngine(s: Script): HarnessEngine {
  const live = new Set<Emit>();
  const emit: Emit = (u) => live.forEach((cb) => cb(u));
  const off = (): (() => void) => () => undefined;
  const unused = (): Promise<never> => Promise.reject(new Error('not used in this test'));
  return {
    kind: 'inline',
    init: async () => ({ gpu: false }) as InitResult,
    processFrame: async (f) => {
      const u = s.frame(f.index!, emit);
      emit(u);
      return u;
    },
    finishCapture: async () => s.finish?.(emit),
    refine: () => Promise.reject(new Error('refine is not scripted')),
    densify: unused,
    mesh: unused,
    splat: unused,
    exportModel: unused,
    cancel: () => undefined,
    onLive: (cb) => {
      live.add(cb);
      return () => live.delete(cb);
    },
    onLog: off,
    onError: off,
    onStatus: off,
    onProgress: off,
    dispose: () => undefined,
  };
}

const update = (frameIndex: number, o: Partial<LiveUpdate> = {}): LiveUpdate => ({
  frameIndex,
  decision: 'skip',
  guidance: 'good',
  tracking: 'tracking',
  numKeyframes: 0,
  numRegistered: 0,
  numPoints: 0,
  timings: {},
  ...o,
});

/** Live camera of keyframe `id` (t = [id, 0, 0] identifies it). */
const cam = (id: number): LiveCamera => ({
  pose: { R: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), t: Float64Array.of(id, 0, 0) },
  intrinsics: { width: 64, height: 48, f: 50, cx: 32, cy: 24, k1: 0, k2: 0 },
});

/** Background model update after a mapping job: the registered keyframes `ids`, sorted. */
const model = (frameIndex: number, numKeyframes: number, ids: number[]): LiveUpdate =>
  update(frameIndex, { numKeyframes, numRegistered: ids.length, numPoints: 100 * ids.length, cameras: ids.map(cam) });

function serveDataset(numFrames: number): void {
  const meta = {
    name: 'stub',
    scene: 'object',
    width: 64,
    height: 48,
    fps: 8,
    numFrames,
    screenAngle: 0,
    sceneExtent: 1,
    intrinsics: { f: 50, cx: 32, cy: 24, k1: 0, k2: 0, width: 64, height: 48 },
  };
  const frames = Array.from({ length: numFrames }, (_, i) => ({ index: i, file: `frame_${i}.jpg`, t: 1000 + 125 * i }));
  vi.stubGlobal('fetch', async (url: string) => {
    const file = url.slice(url.lastIndexOf('/') + 1);
    if (file === 'meta.json') return new Response(JSON.stringify(meta));
    if (file === 'frames.json') return new Response(JSON.stringify(frames));
    if (file.endsWith('.jpg')) return new Response(new Uint8Array(4));
    return new Response('', { status: 404 });
  });
}

const run = (s: Script, config: Partial<HarnessConfig> = {}) => {
  h.engine = stubEngine(s);
  return runPipeline({ dataset: 'stub', baseUrl: '/stub/', engine: 'inline', logFrames: false, ...config }, () => undefined);
};

beforeEach(() => {
  vi.stubGlobal('createImageBitmap', async () => ({ close: () => undefined }));
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('runPipeline live summary', () => {
  it('takes the live cameras from the newest update carrying them, not from later camera-less frame responses', async () => {
    // background mapping: keyframes at deliveries 0, 2, 4, 6; the model update of keyframe j
    // arrives while frame j + 1 is processed, frame responses carry no cameras
    serveDataset(10);
    const kfs = [0, 2, 4, 6];
    const r = await run({
      frame: (k, emit) => {
        const mapped = kfs.filter((x) => x < k);
        if (kfs.includes(k - 1)) emit(model(k - 1, mapped.length, mapped.map((_, i) => i)));
        const n = kfs.filter((x) => x <= k).length;
        return update(k, { decision: kfs.includes(k) ? 'keyframe' : 'skip', numKeyframes: n, numRegistered: mapped.length, numPoints: 100 * mapped.length });
      },
    });
    expect(r.live.unsolicitedUpdates).toBe(4);
    expect(r.live.keyframeIndices).toEqual(kfs);
    expect([r.live.keyframes, r.live.registered]).toEqual([4, 4]);
    expect(r.live.cameras.map((c) => c.pose.t[0])).toEqual([0, 1, 2, 3]);
    expect(r.live.images?.map((x) => [x.index, x.pose.t[0]])).toEqual([
      [0, 0],
      [2, 1],
      [4, 2],
      [6, 3],
    ]);
  });

  describe('keyframes at deliveries 0, 3, 6, 9 and one more committed from the held frame at finishCapture', () => {
    const kfs = [0, 3, 6, 9];
    const frame = (k: number): LiveUpdate => update(k, { decision: kfs.includes(k) ? 'keyframe' : 'skip', numKeyframes: kfs.filter((x) => x <= k).length });

    it('maps no live images when a keyframe is unregistered, even though the camera count matches the announced keyframes', async () => {
      // keyframe id 1 (delivery 3) never registers; the held one (id 4) does
      serveDataset(12);
      const r = await run({ frame, finish: (emit) => emit(model(11, 5, [0, 2, 3, 4])) });
      expect(r.live.keyframeIndices).toEqual(kfs);
      expect([r.live.keyframes, r.live.registered, r.live.cameras.length]).toEqual([5, 4, 4]);
      expect(r.live.images).toBeNull();
    });

    it('maps the live images by keyframe order when every keyframe is registered and announced', async () => {
      serveDataset(12);
      const r = await run({ frame, finish: (emit) => emit(model(11, 4, [0, 1, 2, 3])) });
      expect(r.live.images?.map((x) => [x.index, x.pose.t[0]])).toEqual([
        [0, 0],
        [3, 1],
        [6, 2],
        [9, 3],
      ]);
    });
  });

  it('records the coverage ring of the last live cameras (and of every model update with liveTrace)', async () => {
    // a level orbit (25° elevation) of 36 keyframes, one per delivery, mapped inline
    serveDataset(36);
    const e = (25 * Math.PI) / 180;
    const orbit: LiveCamera[] = Array.from({ length: 36 }, (_, i) => {
      const a = (2 * Math.PI * i) / 36;
      return { pose: lookAt([Math.cos(a) * Math.cos(e), Math.sin(a) * Math.cos(e), Math.sin(e)], [0, 0, 0]), intrinsics: cam(0).intrinsics };
    });
    const r = await run(
      { frame: (k) => update(k, { decision: 'keyframe', numKeyframes: k + 1, numRegistered: k + 1, cameras: orbit.slice(0, k + 1) }) },
      { liveTrace: true, useImu: false },
    );
    const c = r.live.coverage!;
    expect(c.cameras).toBe(36);
    expect(c.stable).toBe(true);
    // no IMU fed ⇒ no gravity ⇒ azimuth only (the app's rule without the motion sensor)
    expect(c.elevation).toBe(false);
    expect(c.coveredSectors).toBe(12);
    expect(c.covered).toEqual([0, 12, 0]);
    expect(c.prompt).toBe('done');
    expect(c.counts[1].every((n) => n === 3)).toBe(true);
    const trace = r.liveTrace!;
    expect(trace).toHaveLength(36);
    expect(trace[2].coverage.stable).toBe(false); // < 6 cameras
    expect(trace[17].coverage.prompt).toBe('other_side'); // half the orbit
    const { counts: _counts, ...summary } = c;
    expect(trace[35].coverage).toEqual(summary);
  });

  it('counts the frame that aborted the capture as delivered', async () => {
    serveDataset(10);
    const r = await run(
      {
        frame: (k) => {
          if (k >= 2) throw new Error(`frame ${k} failed`);
          return update(k);
        },
      },
      { maxFrameErrors: 3 },
    );
    expect(r.frames).toHaveLength(5);
    expect(r.live.delivered).toBe(5);
  });
});
