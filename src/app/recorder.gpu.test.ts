/**
 * The raw capture recorder end to end in Chromium (no GPU work): CaptureRecorder as the FrameTap
 * of a frame stream (VideoFrames, ImageBitmaps, RGBAImages — the originals closed right after the
 * "engine" got them, like a transfer) → the real recorder worker (JPEG via OffscreenCanvas,
 * OPFS sync access handles, or the IndexedDB fallback) → RecordingStore listing → the ZIP-part
 * export → the archive contents (meta, verbatim frames.jsonl, decodable JPEGs at the recorded
 * size and close to the original pixels) → a verbatim harness replay (runPipeline, stubbed fetch
 * over the extracted files, the MockEngine): the FrameInputs reaching the engine equal the
 * recorded ones. Then delete.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { RGBAImage } from '../core/types';
import { decodeFrameRow, frameMeta, parseJsonl, type FrameMeta } from '../io/recording';
import { RecordingStore, prepareRecordingExport } from '../io/recording-store';
import { pumpFrame } from '../io/recording-testdata';
import { readZip } from '../io/zip';
import type { FrameInput, LiveUpdate } from '../workers/protocol';
import { tapFrame } from './frame-tap';
import { MockEngine } from './mock-engine';
import { CaptureRecorder } from './recorder';
import { runPipeline } from '../harness/engine/pipeline';

type Kind = 'VideoFrame' | 'ImageBitmap' | 'RGBAImage';
const KINDS: Kind[] = ['VideoFrame', 'ImageBitmap', 'RGBAImage', 'VideoFrame', 'ImageBitmap', 'VideoFrame'];

/** A test pattern: a horizontal gradient + a frame-specific colour block (so frames differ). */
function pattern(i: number, w: number, h: number): OffscreenCanvas {
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext('2d')!;
  const g = ctx.createLinearGradient(0, 0, w, 0);
  g.addColorStop(0, '#102030');
  g.addColorStop(1, '#e0c080');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = `rgb(${40 * i}, 200, ${255 - 40 * i})`;
  ctx.fillRect(w / 4, h / 4, w / 2, h / 2);
  return c;
}

async function pixels(kind: Kind, c: OffscreenCanvas, i: number): Promise<FrameInput['bitmap']> {
  if (kind === 'VideoFrame') return new VideoFrame(c, { timestamp: i * 100_000 });
  if (kind === 'ImageBitmap') return createImageBitmap(c);
  const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height);
  return { width: d.width, height: d.height, data: d.data } as RGBAImage;
}

const update = (i: number): LiveUpdate => ({ frameIndex: i, decision: i % 2 ? 'skip' : 'keyframe', guidance: 'good', tracking: 'tracking', numKeyframes: 1 + (i >> 1), numRegistered: i >> 1, numPoints: 10 * i, timings: { total: 20 } });

/** Centre pixel of an image (RGB). */
async function centre(src: ImageBitmap | OffscreenCanvas): Promise<number[]> {
  const c = new OffscreenCanvas(src.width, src.height);
  const ctx = c.getContext('2d')!;
  ctx.drawImage(src, 0, 0);
  return Array.from(ctx.getImageData(src.width >> 1, src.height >> 1, 1, 1).data.slice(0, 3));
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe.each(['opfs', 'idb'] as const)('raw capture recording through the real worker (%s)', (prefer) => {
  it('records, lists, exports in ZIP parts, replays verbatim, deletes', async () => {
    const idbName = `sfm-recordings-test-${Math.random().toString(36).slice(2)}`;
    const rec = await CaptureRecorder.start(
      { name: `Test ${prefer}`, sessionId: null, engine: { mode: 'object', quality: 'fast', hfovPriorDeg: 66 }, camera: null, pumpStats: () => ({ skippedBusy: 1 }) },
      { prefer, idbName, available: async () => null },
    );
    expect(rec.sink).toBe(prefer);
    const originals: FrameInput[] = [];
    const sizes: [number, number][] = [];
    for (let i = 0; i < KINDS.length; i++) {
      // frame 3 is larger than the recorder's 1280 px: recorded downscaled
      const [w, h] = i === 3 ? [1600, 1200] : [320, 240];
      const f: FrameInput = { ...pumpFrame(i), bitmap: await pixels(KINDS[i], pattern(i, w, h), i), maxSide: KINDS[i] === 'VideoFrame' ? 1600 : null };
      originals.push(f);
      sizes.push([w, h]);
      // the "engine" closes (transfers) the pixels as soon as it got them
      await tapFrame(rec, f, async (x) => {
        (x.bitmap as { close?: () => void }).close?.();
        return update(i);
      });
    }
    const h = await rec.stop();
    expect(h?.summary).toMatchObject({ frames: KINDS.length, failedImages: 0, pump: { skippedBusy: 1 } });

    const store = await RecordingStore.open({ idbName });
    try {
      const info = (await store.list()).find((r) => r.id === rec.id)!;
      expect(info).toMatchObject({ sink: prefer, header: { state: 'complete', name: `Test ${prefer}` } });
      expect(info.bytes).toBeGreaterThan(KINDS.length * 1000);

      const exp = await prepareRecordingExport(store, info, { maxPartBytes: Math.max(4000, Math.floor(info.bytes / 3)) });
      expect(exp.meta).toMatchObject({ numFrames: KINDS.length, mode: 'object', quality: 'fast', hfovPriorDeg: 66, width: 320, height: 240, missingImages: 0 });
      expect(exp.parts.length).toBeGreaterThan(1);
      const files = new Map<string, Uint8Array>();
      for (const p of exp.parts) {
        const blob = await p.build();
        expect(blob.size).toBe(p.bytes);
        for (const e of readZip(new Uint8Array(await blob.arrayBuffer()))) files.set(e.name, e.data.slice());
      }
      for (const name of ['meta.json', 'recording.json', 'frames.json', 'frames.jsonl', 'imu.json', 'heading.json', 'live.jsonl']) expect(files.has(name), name).toBe(true);
      expect(files.has('geo.json')).toBe(false); // location stripped by default
      const rows = parseJsonl(new TextDecoder().decode(files.get('frames.jsonl')!), decodeFrameRow).rows;
      expect(rows).toHaveLength(KINDS.length);
      for (let i = 0; i < rows.length; i++) {
        const r = rows[i];
        expect(r.input).toStrictEqual({ ...frameMeta(originals[i]), geo: originals[i].geo === undefined ? undefined : null });
        const [w, hh] = sizes[i];
        expect(r.image).toMatchObject({ kind: KINDS[i], srcWidth: w, srcHeight: hh, width: i === 3 ? 1280 : w, height: i === 3 ? 960 : hh });
        const jpeg = files.get(r.image!.file)!;
        expect(r.image!.bytes).toBe(jpeg.length);
        const bmp = await createImageBitmap(new Blob([jpeg as Uint8Array<ArrayBuffer>], { type: 'image/jpeg' }));
        expect([bmp.width, bmp.height]).toEqual([r.image!.width, r.image!.height]);
        const got = await centre(bmp);
        const want = await centre(pattern(i, w, hh));
        for (let k = 0; k < 3; k++) expect(Math.abs(got[k] - want[k]), `frame ${i} channel ${k}`).toBeLessThan(24);
        bmp.close();
      }
      const live = new TextDecoder().decode(files.get('live.jsonl')!).trim().split('\n');
      expect(live).toHaveLength(KINDS.length);

      // verbatim replay of the extracted dataset into the MockEngine
      vi.stubGlobal('fetch', async (url: string) => {
        const f = files.get(url.replace(/^\/rec\//, ''));
        return f ? new Response(f.slice()) : new Response('', { status: 404 });
      });
      const received: FrameMeta[] = [];
      const real = MockEngine.prototype.processFrame;
      vi.spyOn(MockEngine.prototype, 'processFrame').mockImplementation(function (this: MockEngine, f: FrameInput) {
        received.push(frameMeta(f));
        return real.call(this, f);
      });
      const result = await runPipeline({ dataset: 'rec', baseUrl: '/rec/', engine: 'mock', mockSpeed: 200, densify: false, mesh: false, exports: false, logFrames: false }, () => undefined);
      expect(result.errors).toEqual([]);
      expect(received).toHaveLength(KINDS.length);
      for (let i = 0; i < received.length; i++) expect(received[i]).toStrictEqual(rows[i].input);
      expect(result.config).toMatchObject({ verbatim: true, mode: 'object', quality: 'fast', hfovPriorDegUsed: 66 });
      expect(result.recording?.device?.frames).toBe(KINDS.length);

      await store.delete(info);
      expect((await store.list()).some((r) => r.id === rec.id)).toBe(false);
    } finally {
      store.close();
      if (prefer === 'idb') await new Promise((r) => (indexedDB.deleteDatabase(idbName).onsuccess = r));
    }
  });
});
