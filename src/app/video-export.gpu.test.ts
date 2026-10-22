/**
 * Fly-through video on the GPU (headless Chromium): the splat layer's offscreen frames
 * (`Viewer.openSplatFrames`) equal SplatRenderer's own render of the same camera, the canvas and
 * readback VideoFrames hold those pixels (each frame its own, after the session's canvas checks
 * keep it on the canvas path), the viewer does not draw while a session is open (and
 * redraws after), a hidden page holds the next frame back until it is visible again, and a closed
 * session rejects a waiting frame. Then the whole export: path → frames → VideoEncoder → MP4 →
 * VideoDecoder, frame count and PSNR against the directly read frames (skipped where this browser
 * has no H.264 encoder / decoder), and the Diagnostics probe.
 */
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import { pipelineCompileLimits } from '../features/gpu-common';
import { getGPU } from '../gpu/device';
import { readMp4 } from '../io/mp4';
import { SplatRenderer, splatImageToRGBA8 } from '../splat/renderer';
import { lookAt } from '../splat/testutil';
import { SH_C0, allocSplatModel, logit, type SplatCamera, type SplatModel } from '../splat/types';
import { buildFlythrough, flythroughCamera, type FlythroughCamera } from '../viewer/flythrough';
import { Viewer } from '../viewer/viewer';
import { chooseVideoConfig, decodeCount, probeVideoExport, renderFlythroughVideo } from './video-export';

const containers: HTMLElement[] = [];
const viewers: Viewer[] = [];

beforeAll(async () => {
  const c = await getGPU();
  if (!c) throw new Error('WebGPU unavailable in the test browser');
  if (c.isSoftware) pipelineCompileLimits.timeoutMs = 600_000;
}, 600_000);

afterEach(() => {
  for (const v of viewers.splice(0)) v.dispose();
  for (const c of containers.splice(0)) c.remove();
});

function makeViewer(w: number, h: number): Viewer {
  const el = document.createElement('div');
  el.style.cssText = `position:fixed;left:0;top:0;width:${w}px;height:${h}px`;
  document.body.appendChild(el);
  containers.push(el);
  const v = new Viewer(el, { background: '#000000', maxPixelRatio: 1, splatAllowSoftware: true });
  viewers.push(v);
  return v;
}

/** A small coloured cluster of Gaussians around the origin (Z-up world). */
function cluster(): SplatModel {
  const n = 60;
  const m = allocSplatModel(n, 0);
  for (let i = 0; i < n; i++) {
    const a = (i / n) * 2 * Math.PI;
    m.means.set([0.3 * Math.cos(a), 0.3 * Math.sin(a), 0.15 * Math.sin(3 * a)], 3 * i);
    m.logScales.fill(Math.log(0.04), 3 * i, 3 * i + 3);
    m.opacities[i] = logit(0.9);
    m.sh.set([(0.5 + 0.5 * Math.cos(a) - 0.5) / SH_C0, (0.5 + 0.5 * Math.sin(a) - 0.5) / SH_C0, (0.9 - 0.5) / SH_C0], 3 * i);
  }
  return m;
}

function camera(w: number, h: number, eye: [number, number, number] = [0.2, -1.4, 0.6]): SplatCamera {
  const { R, t } = lookAt(eye, [0, 0, 0]);
  return { width: w, height: h, fx: 0.9 * w, fy: 0.9 * w, cx: (w - 1) / 2, cy: (h - 1) / 2, R, t, near: 0.01 };
}

function psnr(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let se = 0;
  let n = 0;
  for (let i = 0; i < a.length; i += 4)
    for (let c = 0; c < 3; c++) {
      se += (a[i + c] - b[i + c]) ** 2;
      n++;
    }
  return se === 0 ? Infinity : 10 * Math.log10((255 * 255) / (se / n));
}

/** RGBA of a VideoFrame drawn into a 2D canvas. */
function framePixels(f: VideoFrame, w: number, h: number): Uint8ClampedArray {
  const c = new OffscreenCanvas(w, h).getContext('2d', { willReadFrequently: true })!;
  c.drawImage(f, 0, 0, w, h);
  return c.getImageData(0, 0, w, h).data;
}

describe('fly-through video (GPU)', () => {
  it('offscreen frames: the renderer’s pixels, canvas and readback VideoFrames, viewer paused, hidden page, close', async () => {
    const W = 96;
    const H = 64;
    const v = makeViewer(120, 90);
    const model = cluster();
    expect(await v.setSplats({ model })).toBe(true);
    const cam = camera(W, H);
    // the same kernels drawn directly by a SplatRenderer (pipelines cached per device)
    const gpu = (await getGPU())!;
    const ref = await SplatRenderer.create(gpu, { shDegree: 0, antialiased: true, maxWidth: W, maxHeight: H });
    ref.setModel(model);
    const direct = splatImageToRGBA8(await ref.render(cam, { background: [0, 0, 0] }));
    ref.destroy();

    const s = v.openSplatFrames({ width: W, height: H })!;
    expect(s).not.toBeNull();
    const framesBefore = v.getStats().splatLayer!.frames;
    v.setOrbitTarget([0.1, 0, 0]);
    v.renderNow();
    await new Promise((r) => setTimeout(r, 50));
    expect(v.getStats().splatLayer!.frames).toBe(framesBefore); // the viewer waits while a session is open
    const px = await s.pixels(cam);
    let maxDiff = 0;
    for (let i = 0; i < px.data.length; i++) if (i % 4 !== 3) maxDiff = Math.max(maxDiff, Math.abs(px.data[i] - direct[i]));
    let lit = 0;
    for (let i = 0; i < px.data.length; i += 4) if (px.data[i] + px.data[i + 1] + px.data[i + 2] > 60) lit++;
    console.log(`[video-gpu] offscreen vs SplatRenderer: max |Δ| ${maxDiff}, lit pixels ${lit}`);
    expect(lit).toBeGreaterThan(200);
    expect(maxDiff).toBeLessThanOrEqual(1);

    // the first frames of an 'auto' session check the canvas path (drawn over: kept, not stale);
    // Chromium passes, so the session stays on it
    const f = await s.videoFrame(cam, 0, 33333);
    console.log(`[video-gpu] frame path ${s.path}${s.fallbackReason ? ` (${s.fallbackReason})` : ''}, format ${f.format}, ${f.codedWidth}×${f.codedHeight}`);
    expect([f.codedWidth, f.codedHeight]).toEqual([W, H]);
    expect(f.timestamp).toBe(0);
    expect(psnr(framePixels(f, W, H), px.data)).toBeGreaterThan(35);
    f.close();
    const cam2 = camera(W, H, [0.4, -1.3, 0.7]);
    const f2 = await s.videoFrame(cam2, 33333, 33333);
    expect(f2.timestamp).toBe(33333);
    const px2 = await s.pixels(cam2);
    // (a different image: frame 1 shown again would fail)
    expect(psnr(px2.data, px.data)).toBeLessThan(30);
    expect(psnr(framePixels(f2, W, H), px2.data)).toBeGreaterThan(35);
    f2.close();
    const f2b = await s.videoFrame(cam, 66667);
    expect(s.path, s.fallbackReason ?? '').toBe('canvas');
    expect(psnr(framePixels(f2b, W, H), px.data)).toBeGreaterThan(35);
    f2b.close();

    // a hidden page: nothing is submitted until it is visible again
    let hidden = true;
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => hidden });
    document.dispatchEvent(new Event('visibilitychange'));
    const n0 = s.frames;
    let done = false;
    const pending = s.pixels(cam).then((p) => ((done = true), p));
    await new Promise((r) => setTimeout(r, 120));
    expect(done).toBe(false);
    expect(s.frames).toBe(n0);
    hidden = false;
    document.dispatchEvent(new Event('visibilitychange'));
    const again = await pending;
    expect(psnr(again.data, px.data)).toBe(Infinity);
    delete (document as unknown as { hidden?: boolean }).hidden;

    // close: a frame waiting for visibility is rejected; the viewer draws again
    hidden = true;
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => hidden });
    const waiting = s.pixels(cam);
    s.close();
    await expect(waiting).rejects.toMatchObject({ name: 'AbortError' });
    delete (document as unknown as { hidden?: boolean }).hidden;
    v.renderNow();
    for (let i = 0; i < 50 && v.getStats().splatLayer!.frames === framesBefore; i++) await new Promise((r) => setTimeout(r, 20));
    expect(v.getStats().splatLayer!.frames).toBeGreaterThan(framesBefore);

    // a forced readback session gives the same image
    const s2 = v.openSplatFrames({ width: W, height: H, path: 'readback' })!;
    const f3 = await s2.videoFrame(cam, 0);
    expect(s2.path).toBe('readback');
    expect(f3.format === 'BGRX' || f3.format === 'RGBX').toBe(true);
    expect(psnr(framePixels(f3, W, H), px.data)).toBeGreaterThan(35);
    f3.close();
    s2.close();
  });

  it('path → frames → H.264 → MP4 → decoded frames (where this browser encodes H.264)', async () => {
    const v = makeViewer(120, 90);
    expect(await v.setSplats({ model: cluster() })).toBe(true);
    // an orbit of 16 photos around the cluster
    const cams: FlythroughCamera[] = Array.from({ length: 16 }, (_, k) => {
      const a = (k / 15) * Math.PI;
      const { R, t } = lookAt([1.2 * Math.cos(a), 1.2 * Math.sin(a), 0.5], [0, 0, 0]);
      return { pose: { R: Float64Array.from(R), t: Float64Array.from(t) }, intrinsics: { width: 480, height: 640, f: 500 } };
    });
    const path = buildFlythrough(cams, { longSide: 192, shortSide: 128, minSeconds: 1, maxSeconds: 1 })!;
    expect([path.width, path.height, path.poses.length]).toEqual([128, 192, 30]);
    const config = await chooseVideoConfig(path.width, path.height);
    if (!config) {
      // Playwright's Chromium has no H.264 encoder: the same loop with a real VP8 encoder, its
      // chunks muxed under a stand-in avcC (structure only) and decoded by a VP8 VideoDecoder
      console.warn('[video-gpu] no H.264 encoder in this browser: VP8 round trip instead');
      const vp8: VideoEncoderConfig = { codec: 'vp8', width: path.width, height: path.height, bitrate: 6_000_000, framerate: 30 };
      expect((await VideoEncoder.isConfigSupported(vp8)).supported).toBe(true);
      const standIn = Uint8Array.from([1, 0x42, 0xe0, 0x1f, 0xff, 0xe1, 0, 4, 0x67, 0x42, 0xe0, 0x1f, 1, 0, 2, 0x68, 0xce]);
      const frames = v.openSplatFrames({ width: path.width, height: path.height })!;
      try {
        const res = await renderFlythroughVideo({
          frames,
          path,
          config: vp8,
          createEncoder: (init) => new VideoEncoder({ output: (chunk) => init.output(chunk, { decoderConfig: { codec: 'avc1.42e01f', description: standIn } }), error: init.error }),
        });
        console.log(`[video-gpu] VP8: ${res.frames} frames via ${res.framePath}: ${res.blob.size} B, ${res.msPerFrame.toFixed(1)} ms per frame (GPU ${res.renderMsPerFrame.toFixed(1)} ms)`);
        expect(res.maxRawFrames).toBe(1);
        const bytes = new Uint8Array(await res.blob.arrayBuffer());
        const info = readMp4(bytes);
        expect(info.samples.length).toBe(30);
        expect(info.stss).toEqual([1]); // one key frame per 2 s
        const decoded: VideoFrame[] = [];
        const dec = new VideoDecoder({ output: (f) => decoded.push(f), error: (e) => console.error(e) });
        dec.configure({ codec: 'vp8', codedWidth: path.width, codedHeight: path.height });
        for (const s of info.samples) dec.decode(new EncodedVideoChunk({ type: s.key ? 'key' : 'delta', timestamp: (s.pts * 1e6) / info.timescale, data: bytes.subarray(s.offset, s.offset + s.size) }));
        await dec.flush();
        dec.close();
        expect(decoded.length).toBe(30);
        const psnrs: number[] = [];
        for (let i = 0; i < 30; i += 7) {
          const px = await frames.pixels(flythroughCamera(path, i));
          psnrs.push(psnr(framePixels(decoded[i], path.width, path.height), px.data));
        }
        for (const f of decoded) f.close();
        console.log(`[video-gpu] VP8 decoded vs direct PSNR ${psnrs.map((p) => p.toFixed(1)).join(' ')} dB`);
        expect(Math.min(...psnrs)).toBeGreaterThan(30);
      } finally {
        frames.close();
      }
      return;
    }
    const frames = v.openSplatFrames({ width: path.width, height: path.height })!;
    try {
      const res = await renderFlythroughVideo({ frames, path, config });
      console.log(`[video-gpu] ${res.frames} frames ${res.codec} via ${res.framePath}: ${res.blob.size} B, ${res.msPerFrame.toFixed(1)} ms per frame (GPU ${res.renderMsPerFrame.toFixed(1)} ms)`);
      expect(res.maxRawFrames).toBe(1);
      const bytes = new Uint8Array(await res.blob.arrayBuffer());
      const info = readMp4(bytes);
      expect(info.samples.length).toBe(30);
      expect([info.width, info.height]).toEqual([128, 192]);
      expect(info.samples[0].key).toBe(true);
      // decoded vs the directly read frames (encode loss only)
      const psnrs: number[] = [];
      const n = await decodeCount(bytes, async (f, i) => {
        if (i % 7 !== 0) return;
        const px = await frames.pixels(flythroughCamera(path, i));
        psnrs.push(psnr(framePixels(f, path.width, path.height), px.data));
      });
      if (n === null) console.warn('[video-gpu] no H.264 decoder in this browser: decode check skipped');
      else {
        console.log(`[video-gpu] decoded ${n} frames, PSNR vs direct ${psnrs.map((p) => p.toFixed(1)).join(' ')} dB`);
        expect(n).toBe(30);
        expect(Math.min(...psnrs)).toBeGreaterThan(30);
      }
    } finally {
      frames.close();
    }
  });

  it('the Diagnostics probe reports a codec, a frame path and the encode time', async () => {
    const rep = await probeVideoExport(12);
    console.log(`[video-gpu] probe ${JSON.stringify(rep)}`);
    expect(rep.webCodecs).toBe(true);
    // the frame path is probed with or without an encoder (Chromium: the canvas path, its frames
    // kept once the canvas is drawn over and not stale)
    expect(rep.framePath, rep.canvasNote ?? '').toBe('canvas');
    if (!rep.codecPortrait) {
      expect(rep.error).toMatch(/H\.264/);
      return;
    }
    expect(rep.error).toBeNull();
    expect(rep.frames).toBe(12);
    expect(rep.msPerFrame).toBeGreaterThan(0);
    if (rep.decoded !== null) expect([rep.decoded, rep.decodedColours]).toEqual([12, true]);
  });
});
