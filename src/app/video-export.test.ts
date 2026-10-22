/**
 * Video export logic without a browser (src/app/video-export.ts): the H.264 config ladder, the
 * encode loop against a fake VideoEncoder and frame source — one raw frame alive at a time, key
 * frames every 2 s, chunks muxed into a valid MP4, cancel and encoder errors — and the camera path
 * of a sparse model (cameras sorted into capture order).
 */
import { afterEach, describe, expect, it } from 'vitest';
import { readMp4 } from '../io/mp4';
import { buildFlythrough, type FlythroughPath } from '../viewer/flythrough';
import { OBJECT_REAL } from '../viewer/flythrough.testdata';
import type { SparseResult } from '../workers/protocol';
import { chooseVideoConfig, flythroughForModel, renderFlythroughVideo, VIDEO_CODECS, type FlythroughFrameSource } from './video-export';

const FAKE_AVCC = Uint8Array.from([1, 0x42, 0xe0, 0x1f, 0xff, 0xe1, 0x00, 0x04, 0x67, 0x42, 0xe0, 0x1f, 0x01, 0x00, 0x02, 0x68, 0xce]);

const g = globalThis as unknown as { VideoEncoder?: unknown };
afterEach(() => {
  delete g.VideoEncoder;
});

/** A fake VideoEncoder: encodes asynchronously (queue drains after a tick), first output carries the avcC. */
function fakeEncoderFactory(log: { keys: number[]; configured: VideoEncoderConfig | null; closed: boolean; failAt?: number }) {
  return (init: VideoEncoderInit): VideoEncoder => {
    let queue = 0;
    let state: CodecState = 'unconfigured';
    let n = 0;
    const pending: Promise<void>[] = [];
    const enc = {
      get encodeQueueSize() {
        return queue;
      },
      get state() {
        return state;
      },
      configure(c: VideoEncoderConfig) {
        log.configured = c;
        state = 'configured';
      },
      encode(frame: VideoFrame, opts?: VideoEncoderEncodeOptions) {
        if (state !== 'configured') throw new DOMException('not configured', 'InvalidStateError');
        const i = n++;
        const ts = frame.timestamp;
        const key = !!opts?.keyFrame;
        if (key) log.keys.push(i);
        queue++;
        pending.push(
          new Promise<void>((r) => setTimeout(r, 1)).then(() => {
            queue--;
            if (log.failAt === i) {
              state = 'closed';
              init.error(new DOMException('Codec reclaimed due to inactivity', 'QuotaExceededError'));
              return;
            }
            const data = Uint8Array.from([0, 0, 0, 2, key ? 0x65 : 0x41, i & 0xff]);
            const chunk = { type: key ? 'key' : 'delta', timestamp: ts, byteLength: data.length, copyTo: (dst: Uint8Array) => dst.set(data) } as unknown as EncodedVideoChunk;
            init.output(chunk, i === 0 ? { decoderConfig: { codec: 'avc1.42e01f', description: FAKE_AVCC } } : {});
          }),
        );
      },
      async flush() {
        await Promise.all(pending);
      },
      close() {
        state = 'closed';
        log.closed = true;
      },
    };
    return enc as unknown as VideoEncoder;
  };
}

/** A frame source that counts live frames. */
function fakeFrames(): FlythroughFrameSource & { alive: number; maxAlive: number; made: number[] } {
  const src = {
    path: 'readback' as const,
    totalFrameMs: 0,
    frames: 0,
    alive: 0,
    maxAlive: 0,
    made: [] as number[],
    async videoFrame(_cam: unknown, timestampUs: number) {
      src.alive++;
      src.maxAlive = Math.max(src.maxAlive, src.alive);
      src.made.push(timestampUs);
      src.frames++;
      src.totalFrameMs += 5;
      return { timestamp: timestampUs, close: () => void src.alive-- } as unknown as VideoFrame;
    },
  };
  return src;
}

function testPath(): FlythroughPath {
  const cams = OBJECT_REAL.poses.map((p) => ({ pose: { R: Float64Array.from(p.slice(0, 9)), t: Float64Array.from(p.slice(9, 12)) }, intrinsics: OBJECT_REAL.intrinsics }));
  return buildFlythrough(cams, { maxSeconds: 6 })!;
}

describe('video export', () => {
  it('chooseVideoConfig walks the H.264 ladder High → Main → Baseline, avc format, 30 fps 6 Mbit/s', async () => {
    expect(await chooseVideoConfig(720, 1280)).toBeNull(); // no WebCodecs in Node
    const asked: string[] = [];
    let supported = new Set<string>(['avc1.42e01f']);
    g.VideoEncoder = {
      isConfigSupported: async (c: VideoEncoderConfig) => {
        asked.push(c.codec);
        return { supported: supported.has(c.codec), config: c };
      },
    };
    const c = await chooseVideoConfig(720, 1280);
    expect(asked).toEqual([...VIDEO_CODECS]);
    expect(c).toMatchObject({ codec: 'avc1.42e01f', width: 720, height: 1280, framerate: 30, bitrate: 6_000_000, avc: { format: 'avc' } });
    supported = new Set(['avc1.640028', 'avc1.42e01f']);
    expect((await chooseVideoConfig(1280, 720))?.codec).toBe('avc1.640028');
    supported = new Set();
    expect(await chooseVideoConfig(1280, 720)).toBeNull();
  });

  it('encodes one raw frame at a time, key frames every 2 s, into a valid MP4', async () => {
    const path = testPath();
    const N = path.poses.length;
    expect(N).toBe(180);
    const log = { keys: [] as number[], configured: null as VideoEncoderConfig | null, closed: false };
    const frames = fakeFrames();
    const progress: number[] = [];
    const res = await renderFlythroughVideo({
      frames,
      path,
      config: { codec: 'avc1.42e01f', width: path.width, height: path.height },
      createEncoder: fakeEncoderFactory(log),
      onProgress: (d) => progress.push(d),
    });
    expect(log.configured?.codec).toBe('avc1.42e01f');
    expect(log.closed).toBe(true);
    expect(frames.maxAlive).toBe(1);
    expect(res.maxRawFrames).toBe(1);
    expect(log.keys).toEqual([0, 60, 120]);
    expect(progress.length).toBe(N);
    expect(frames.made.slice(0, 4)).toEqual([0, 33333, 66667, 100000]);
    expect(res).toMatchObject({ frames: N, width: 720, height: 1280, seconds: 6, codec: 'avc1.42e01f', framePath: 'readback', renderMsPerFrame: 5 });
    const info = readMp4(new Uint8Array(await res.blob.arrayBuffer()));
    expect(info.samples.length).toBe(N);
    expect(info.stss).toEqual([1, 61, 121]);
    expect(Array.from(info.avcC)).toEqual(Array.from(FAKE_AVCC));
    expect(info.samples.map((s) => s.pts)).toEqual(Array.from({ length: N }, (_, i) => i * 1000));
  });

  it('a cancel stops the loop and closes the encoder; an encoder error fails the export', async () => {
    const path = testPath();
    const ctl = new AbortController();
    const log = { keys: [] as number[], configured: null as VideoEncoderConfig | null, closed: false };
    const frames = fakeFrames();
    const run = renderFlythroughVideo({
      frames,
      path,
      config: { codec: 'avc1.42e01f', width: path.width, height: path.height },
      createEncoder: fakeEncoderFactory(log),
      signal: ctl.signal,
      onProgress: (d) => {
        if (d === 10) ctl.abort();
      },
    });
    await expect(run).rejects.toMatchObject({ name: 'AbortError' });
    expect(frames.made.length).toBe(10);
    expect(log.closed).toBe(true);
    expect(frames.alive).toBe(0);
    const log2 = { keys: [] as number[], configured: null as VideoEncoderConfig | null, closed: false, failAt: 30 };
    await expect(
      renderFlythroughVideo({ frames: fakeFrames(), path, config: { codec: 'avc1.42e01f', width: path.width, height: path.height }, createEncoder: fakeEncoderFactory(log2) }),
    ).rejects.toMatchObject({ name: 'QuotaExceededError' });
  });

  it('flythroughForModel sorts the cameras by id and uses +Z up when gravity aligned', () => {
    const cams = OBJECT_REAL.poses.map((p, i) => ({ id: i, pose: { R: Float64Array.from(p.slice(0, 9)), t: Float64Array.from(p.slice(9, 12)) }, intrinsics: { ...OBJECT_REAL.intrinsics, k1: 0, k2: 0 } }));
    const shuffled = [...cams.slice(25), ...cams.slice(0, 25)];
    const pts = new Float32Array(0);
    const sparse = { cameras: shuffled, metric: { gravityAligned: true }, points: { count: 0, positions: pts, colors: new Uint8Array(0) } } as unknown as SparseResult;
    const a = flythroughForModel(sparse)!;
    const b = buildFlythrough(cams, { fps: 30 })!;
    expect(a.mode).toBe('object');
    expect(a.poses.length).toBe(b.poses.length);
    expect(Array.from(a.poses[50].t)).toEqual(Array.from(b.poses[50].t));
    expect(flythroughForModel({ ...sparse, cameras: cams.slice(0, 1) } as SparseResult)).toBeNull();
  });
});
