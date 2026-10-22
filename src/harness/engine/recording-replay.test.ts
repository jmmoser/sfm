/**
 * Round trip of a raw capture recording (Node): a synthetic FrameInput sequence goes through the
 * recorder's writer (RecordingWriter, src/io/recording.ts) into an in-memory sink, is exported as
 * the e2e dataset in ZIP parts (planZipParts → createZip → the Node importer's reader
 * e2e/lib/zip-read.mjs), served to runPipeline by a stubbed fetch, and replayed into a stub
 * engine: the FrameInputs that reach it must be IDENTICAL to the recorded ones (every field,
 * typed arrays, −∞ / −0 / NaN), in the recorded order, with the recorded EngineOptions.
 * The pixels are a fake "JPEG" (a size + tag header) decoded by a stubbed createImageBitmap.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { computeSelfMetrics } from '../../../e2e/lib/self-metrics.mjs';
import { readStoreZip } from '../../../e2e/lib/zip-read.mjs';
import { RecordingWriter, buildDatasetFiles, frameMeta, planZipParts, zipEntryBytes, type EncodedImage, type FrameMeta } from '../../io/recording';
import { MemorySink, header, pumpFrame } from '../../io/recording-testdata';
import { createZip } from '../../io/zip';
import type { RGBAImage } from '../../core/types';
import type { EngineOptions, FrameInput, InitResult, LiveUpdate } from '../../workers/protocol';
import type { HarnessEngine } from './engines';
import type { HarnessConfig } from './types';

const h = vi.hoisted(() => ({ engine: null as unknown }));
vi.mock('./engines', () => ({ createHarnessEngine: async () => h.engine }));

import { runPipeline } from './pipeline';

const N = 9;
/** The frame whose pixels "failed" on the device (no image: left out of the replay). */
const NO_PIXELS = 4;
const decision = (i: number): LiveUpdate['decision'] => (i % 3 === 0 ? 'keyframe' : 'skip');

/** Fake JPEG: 0xFF 0xD8, width, height (u16 BE), tag. */
function fakeJpeg(w: number, h: number, tag: number): EncodedImage {
  return { jpeg: Uint8Array.of(0xff, 0xd8, w >> 8, w & 255, h >> 8, h & 255, tag), width: w, height: h };
}

interface Received {
  meta: FrameMeta;
  width: number;
  height: number;
  tag: number;
}

/** A HarnessEngine that keeps every FrameInput it gets (pixels as their decoded size + tag). */
function capturingEngine(received: Received[], inits: EngineOptions[]): HarnessEngine {
  const off = (): (() => void) => () => undefined;
  const unused = (): Promise<never> => Promise.reject(new Error('not used in this test'));
  return {
    kind: 'inline',
    init: async (o) => {
      inits.push(o);
      return { gpu: false } as InitResult;
    },
    processFrame: async (f: FrameInput) => {
      const b = f.bitmap as unknown as { width: number; height: number; tag: number };
      received.push({ meta: frameMeta(f), width: b.width, height: b.height, tag: b.tag });
      const i = f.index ?? -1;
      return { frameIndex: i, decision: decision(i), guidance: 'good', tracking: 'tracking', numKeyframes: 0, numRegistered: 0, numPoints: 0, timings: {} };
    },
    finishCapture: async () => undefined,
    refine: () => Promise.reject(new Error('refine is not scripted')),
    densify: unused,
    mesh: unused,
    splat: unused,
    exportModel: unused,
    cancel: () => undefined,
    onLive: off,
    onLog: off,
    onError: off,
    onStatus: off,
    onProgress: off,
    dispose: () => undefined,
  };
}

/** Record N frames through the writer, export them as ZIP parts and extract those (path → bytes). */
async function recordAndExport(maxPartBytes: number): Promise<{ files: Map<string, Uint8Array>; originals: FrameInput[]; parts: number }> {
  const sink = new MemorySink();
  const w = new RecordingWriter(sink);
  await w.writeHeader(header());
  const originals: FrameInput[] = [];
  for (let i = 0; i < N; i++) {
    const f = pumpFrame(i);
    originals.push(f);
    const rgba = f.bitmap as RGBAImage;
    const img = i === NO_PIXELS ? Promise.reject(new Error('convertToBlob failed')) : fakeJpeg(rgba.width - i, rgba.height, i);
    await w.writeFrame({ seq: i, sentAt: 10_000 + i, input: frameMeta(f), kind: 'RGBAImage', srcWidth: rgba.width, srcHeight: rgba.height }, img);
    const upd: LiveUpdate = { frameIndex: i, decision: decision(i), guidance: 'good', tracking: 'tracking', numKeyframes: 1 + Math.floor(i / 3), numRegistered: Math.floor(i / 3), numPoints: 5 * i, timings: { total: 40 } };
    await w.writeLive({ seq: i, index: i, ms: 40 + i, waitMs: i === 7 ? 12 : 0, update: upd, error: null });
  }
  await w.close(header({ state: 'complete', stoppedAt: 1, stopReason: 'capture ended', summary: { frames: 0, imageBytes: 0, failedImages: 0, waitMs: 12, durationMs: 900, pump: { skippedBusy: 3 } } }));
  const b = buildDatasetFiles(JSON.parse(sink.header), sink.text('frames.jsonl'), sink.text('live.jsonl'), { stripLocation: false });
  const data = new Map<string, Uint8Array | string>();
  for (const t of b.texts) data.set(t.name, t.text);
  for (const im of b.images) data.set(im.name, sink.images.get((im.image.store as { key: string }).key)!);
  const enc = new TextEncoder();
  const sizes = [...data].map(([name, d]) => ({ name, size: typeof d === 'string' ? enc.encode(d).length : d.length }));
  const plan = planZipParts(sizes, Math.max(maxPartBytes, ...sizes.map((s) => zipEntryBytes(s.name, s.size) + 22)));
  const files = new Map<string, Uint8Array>();
  for (const p of plan) for (const e of readStoreZip(createZip(p.names.map((n) => ({ name: n, data: data.get(n)! }))))) files.set(e.name, e.data.slice());
  return { files, originals, parts: plan.length };
}

function serve(files: Map<string, Uint8Array>): void {
  vi.stubGlobal('fetch', async (url: string) => {
    const rel = url.replace(/^\/rec\//, '');
    const f = files.get(rel);
    return f ? new Response(f.slice()) : new Response('', { status: 404 });
  });
  vi.stubGlobal('createImageBitmap', async (blob: Blob) => {
    const b = new Uint8Array(await blob.arrayBuffer());
    if (b[0] !== 0xff || b[1] !== 0xd8) throw new Error('not a fake JPEG');
    return { width: (b[2] << 8) | b[3], height: (b[4] << 8) | b[5], tag: b[6], close: () => undefined };
  });
}

async function replay(config: Partial<HarnessConfig> = {}, maxPartBytes = 600) {
  const { files, originals, parts } = await recordAndExport(maxPartBytes);
  serve(files);
  const received: Received[] = [];
  const inits: EngineOptions[] = [];
  h.engine = capturingEngine(received, inits);
  const result = await runPipeline({ dataset: 'rec', baseUrl: '/rec/', engine: 'inline', logFrames: false, ...config }, () => undefined);
  return { result, received, inits, originals, files, parts };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('raw capture recording → verbatim replay', () => {
  it('the engine receives exactly the recorded FrameInputs, in order, with the recorded EngineOptions', async () => {
    const { result, received, inits, originals, parts } = await replay();
    expect(parts).toBeGreaterThan(2); // the export really went through several ZIP parts
    const kept = originals.filter((_, i) => i !== NO_PIXELS);
    expect(received).toHaveLength(kept.length);
    for (let k = 0; k < kept.length; k++) {
      expect(received[k].meta).toStrictEqual(frameMeta(kept[k]));
      const i = kept[k].index!;
      expect([received[k].width, received[k].height, received[k].tag]).toEqual([64 - i, 48, i]);
    }
    // the awkward values survived the whole path
    expect(received[0].meta.screenAngleSince).toBe(-Infinity);
    expect(received[0].meta.gravityDevice).toBeInstanceOf(Float64Array);
    expect(Object.is(received[0].meta.imu[0].gyro[1], -0)).toBe(true);
    expect(received[1].meta.imu[1].gyro[2]).toBeNaN();
    // the recorded EngineOptions are the defaults of the replay
    expect(inits[0]).toMatchObject({ mode: 'object', quality: 'balanced', hfovPriorDeg: 67.3 });
    expect(result.config).toMatchObject({ verbatim: true, mode: 'object', quality: 'balanced', hfovPriorDegUsed: 67.3, captureDateMs: header().createdAt });
    expect(result.dataset.recording).toBe(true);
    // the device's answers (live.jsonl) against the replay's: identical stub decisions
    expect(result.recording).toMatchObject({ verbatim: true, rows: N - 1, skipped: 1, agreement: { compared: N - 1, same: N - 1, rate: 1 } });
    expect(result.recording?.device).toMatchObject({ frames: N, keyframes: 1 + Math.floor((N - 1) / 3), errors: 0 });
    expect(result.recording?.device?.waitMs.max).toBe(12);
    // frame records / keyframe indices use the recorded indices
    expect(result.frames.map((f) => f.index)).toEqual(kept.map((f) => f.index));
    expect(result.live.keyframeIndices).toEqual(kept.filter((f) => decision(f.index!) === 'keyframe').map((f) => f.index));
    // GT-free scoring of the (refine-less) result does not crash
    const m = computeSelfMetrics(result, { name: 'rec', scene: 'recording' });
    expect(m.groundTruth).toBe(false);
    expect(m.registration.delivered).toBe(N - 1);
    expect(m.agreement?.rate).toBe(1);
  });

  it('with a stride the skipped rows’ IMU samples go to the next replayed frame (like the FramePump drains them)', async () => {
    const { received, originals } = await replay({ frameStride: 2 });
    const kept = originals.filter((_, i) => i !== NO_PIXELS);
    // positions 0, 2, 4, 6 of the 8 frames with pixels
    expect(received.map((r) => r.meta.index)).toEqual([0, 2, 4, 6].map((p) => kept[p].index));
    expect(received[1].meta.imu).toEqual([...kept[1].imu, ...kept[2].imu]);
    expect(received[2].meta.imu).toEqual([...kept[3].imu, ...kept[4].imu]);
    // everything else of the frame verbatim
    const { imu: _i, ...rest } = received[1].meta;
    const { imu: _j, ...orig } = frameMeta(kept[2]);
    expect(rest).toStrictEqual(orig);
  });

  it('withholding a sensor clears it the way the app would without it', async () => {
    const { received } = await replay({ useGps: false, useHeading: false, useImu: false });
    for (const r of received) {
      expect(r.meta.geo).toBeNull();
      expect(r.meta.heading).toBeNull();
      expect(r.meta.attitude).toBeNull();
      expect(r.meta.imu).toEqual([]);
      expect(r.meta.imuCalibrationVersion).toBe(-1);
      expect(r.meta.gravityDevice).toBeNull();
    }
  });

  it('verbatim: false synthesizes the FrameInputs from frames.json / imu.json instead', async () => {
    const { received, result } = await replay({ verbatim: false });
    expect(result.recording?.verbatim).toBe(false);
    expect(result.recording?.agreement).toBeNull();
    expect(received.map((r) => r.meta.index)).toEqual(received.map((_, k) => k));
    expect(received.every((r) => r.meta.timeSource === 'captureTime' && r.meta.segment === 0)).toBe(true);
  });
});
