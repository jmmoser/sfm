import { describe, expect, it } from 'vitest';
import type { FrameInput } from '../workers/protocol';
import { MockEngine } from './mock-engine';

type Rgb = [number, number, number];

/** A frame of one colour as the FramePump delivers it: an ImageBitmap or a VideoFrame. */
async function solidFrame(kind: 'ImageBitmap' | 'VideoFrame', rgb: Rgb, t: number): Promise<FrameInput> {
  const c = new OffscreenCanvas(96, 72);
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = `rgb(${rgb.join(',')})`;
  ctx.fillRect(0, 0, c.width, c.height);
  const bitmap = kind === 'ImageBitmap' ? await createImageBitmap(c) : new VideoFrame(c, { timestamp: Math.round(t * 1000) });
  return { bitmap, t, screenAngle: 0, imu: [] };
}

const closed = (f: FrameInput): boolean => (f.bitmap instanceof VideoFrame ? f.bitmap.format === null : (f.bitmap as ImageBitmap).width === 0);

describe('MockEngine frames', () => {
  // Without the read, createImageBitmap(<video>) copies queued up in the GPU process without
  // back-pressure (headless: 100–230 MB/s, screenshots waiting minutes; mock-engine.ts header).
  it.each(['ImageBitmap', 'VideoFrame'] as const)("reads every %s's pixels before closing it, like the engine", async (kind) => {
    const e = new MockEngine({ speed: 8, sessions: async () => null });
    await e.init({ mode: 'object', quality: 'fast' });
    const colours: Rgb[] = [
      [255, 0, 0],
      [0, 0, 255],
      [255, 255, 255],
      [0, 0, 0],
    ];
    for (let i = 0; i < colours.length; i++) {
      const f = await solidFrame(kind, colours[i], 1000 + 100 * i);
      await e.processFrame(f);
      expect(closed(f)).toBe(true);
      const s = e.lastFrameSample;
      expect(s).not.toBeNull();
      for (let k = 0; k < 3; k++) expect(Math.abs(s![k] - colours[i][k])).toBeLessThanOrEqual(2);
    }
    e.dispose();
  });

  it('a frame that cannot be read (already closed) is still processed', async () => {
    const e = new MockEngine({ speed: 8, sessions: async () => null });
    await e.init({ mode: 'object', quality: 'fast' });
    const f = await solidFrame('ImageBitmap', [10, 200, 30], 1000);
    (f.bitmap as ImageBitmap).close();
    const u = await e.processFrame(f);
    expect(u.frameIndex).toBe(0);
    expect(e.lastFrameSample).toBeNull();
    e.dispose();
  });
});
