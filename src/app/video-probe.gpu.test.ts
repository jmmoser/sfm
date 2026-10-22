/**
 * The content checks of the Diagnostics video probe (`probeVideoExport`, src/app/video-export.ts)
 * in Chromium with a stand-in codec, since Playwright's Chromium has no H.264 encoder: the
 * "encoder" stores each frame's centre colour in its chunk and the "decoder" paints it back. With
 * every frame in its own colour the probe passes. An encoder source that keeps showing the first
 * frame (a stale snapshot) or the previous one (a frame read after the next was drawn) is reported
 * with the frame that went wrong. The frame path check itself runs on the real WebGPU canvas.
 * (Own file: VideoEncoder / VideoDecoder are replaced for the page.)
 */
import { afterAll, describe, expect, it } from 'vitest';
import { probeColour, probeVideoExport } from './video-export';

/** A stand-in avcC (SPS / PPS of High 4.0) so that the MP4 parses. */
const STAND_IN = Uint8Array.from([1, 0x64, 0x00, 0x28, 0xff, 0xe1, 0, 4, 0x67, 0x64, 0, 0x28, 1, 0, 2, 0x68, 0xce]);
const RealEncoder = VideoEncoder;
const RealDecoder = VideoDecoder;
let source: 'own' | 'first' | 'previous' = 'own';

function centre(f: VideoFrame): number[] {
  const c = new OffscreenCanvas(8, 8).getContext('2d', { willReadFrequently: true })!;
  c.drawImage(f, 0, 0, 8, 8);
  return Array.from(c.getImageData(4, 4, 1, 1).data.subarray(0, 3));
}

/** Emits each frame's centre colour (or the first / previous frame's, per `source`) as the chunk. */
class ColourEncoder {
  encodeQueueSize = 0;
  state: CodecState = 'unconfigured';
  private n = 0;
  private first: number[] | null = null;
  private prev: number[] | null = null;
  constructor(private readonly init: VideoEncoderInit) {}
  static async isConfigSupported(c: VideoEncoderConfig): Promise<VideoEncoderSupport> {
    return { supported: c.codec.startsWith('avc1'), config: c };
  }
  configure(): void {
    this.state = 'configured';
  }
  encode(f: VideoFrame, o?: VideoEncoderEncodeOptions): void {
    const px = centre(f);
    this.first ??= px;
    const shown = source === 'first' ? this.first : source === 'previous' ? (this.prev ?? px) : px;
    this.prev = px;
    const key = !!o?.keyFrame;
    const i = this.n++;
    const data = Uint8Array.from([0, 0, 0, 4, key ? 0x65 : 0x41, ...shown]);
    this.init.output(new EncodedVideoChunk({ type: key ? 'key' : 'delta', timestamp: f.timestamp, data }), i === 0 ? { decoderConfig: { codec: 'avc1.640028', description: STAND_IN } } : {});
  }
  async flush(): Promise<void> {}
  close(): void {
    this.state = 'closed';
  }
}

/** Paints a 16 × 16 frame in the chunk's colour. */
class ColourDecoder {
  state: CodecState = 'unconfigured';
  constructor(private readonly init: VideoDecoderInit) {}
  static async isConfigSupported(c: VideoDecoderConfig): Promise<VideoDecoderSupport> {
    return { supported: c.codec.startsWith('avc1'), config: c };
  }
  configure(): void {
    this.state = 'configured';
  }
  decode(chunk: EncodedVideoChunk): void {
    const b = new Uint8Array(chunk.byteLength);
    chunk.copyTo(b);
    const data = new Uint8Array(4 * 16 * 16);
    for (let k = 0; k < 256; k++) data.set([b[5], b[6], b[7], 255], 4 * k);
    this.init.output(new VideoFrame(data, { format: 'RGBX', codedWidth: 16, codedHeight: 16, timestamp: chunk.timestamp }));
  }
  async flush(): Promise<void> {}
  close(): void {
    this.state = 'closed';
  }
}

const g = globalThis as unknown as { VideoEncoder: unknown; VideoDecoder: unknown };
g.VideoEncoder = ColourEncoder;
g.VideoDecoder = ColourDecoder;
afterAll(() => {
  g.VideoEncoder = RealEncoder;
  g.VideoDecoder = RealDecoder;
});

describe('video probe content checks', () => {
  it('consecutive probe colours differ by ≥ 89 levels in red', () => {
    for (let i = 1; i < 30; i++) expect(Math.abs(probeColour(i)[0] - probeColour(i - 1)[0]) * 255).toBeGreaterThanOrEqual(89);
  });

  it('every frame in its own colour passes; a stale or late source is reported', async () => {
    source = 'own';
    const ok = await probeVideoExport(9);
    console.log(`[video-probe] ${JSON.stringify(ok)}`);
    // the WebGPU canvas keeps a frame's image once drawn over, and a new frame is not stale (Chromium)
    expect(ok.framePath, ok.canvasNote ?? '').toBe('canvas');
    expect(ok.canvasNote).toBeNull();
    expect(ok.error).toBeNull();
    expect([ok.frames, ok.decoded, ok.decodedColours, ok.decodeNote]).toEqual([9, 9, true, null]);

    source = 'first';
    const stale = await probeVideoExport(9);
    expect([stale.decoded, stale.decodedColours]).toEqual([9, false]);
    expect(stale.decodeNote).toMatch(/^8 of 9 decoded frames show the wrong colour; frame 1 shows/);

    source = 'previous';
    const late = await probeVideoExport(9);
    expect(late.decodedColours).toBe(false);
    expect(late.decodeNote).toMatch(/frame 1 shows/);
  });
});
