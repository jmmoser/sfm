/**
 * The Export sheet's video row (createVideoExport, src/app/video-export.ts) as a state machine, in
 * Chromium with a fake host and frame source (CPU VideoFrames, no GPU). VideoEncoder is replaced
 * by a stub that reports H.264 support and emits small chunks with a stand-in avcC (Playwright's
 * Chromium has no H.264 encoder, and its software VP8 takes ~0.5 s per 720p frame on a loaded
 * host; the real encoder loop is covered by video-export.gpu.test.ts). The MP4 is checked for
 * structure only.
 * Covered: Prepare → progress (updated in place: the row is not rebuilt) → Save (download);
 * Cancel; a hidden page stops it; a photoreal build (cancel while another job holds app.busy)
 * leaves busy and the wake lock to it; dispose mid-export lets them go at once. (Own file: the
 * encoder support is cached per page.)
 */
import { afterAll, describe, expect, it } from 'vitest';
import { readMp4 } from '../io/mp4';
import { lookAt } from '../splat/testutil';
import type { SplatOffscreen } from '../viewer/splat-layer';
import type { SparseResult, SplatResult } from '../workers/protocol';
import type { App, Results } from './app';
import { createVideoExport, type VideoExportController } from './video-export';

const STAND_IN = Uint8Array.from([1, 0x64, 0x00, 0x28, 0xff, 0xe1, 0, 4, 0x67, 0x64, 0, 0x28, 1, 0, 2, 0x68, 0xce]);
const RealEncoder = VideoEncoder;

/** Takes each frame after a tick and emits a 6-byte chunk (the first with the stand-in avcC). */
class StubEncoder {
  encodeQueueSize = 0;
  state: CodecState = 'unconfigured';
  private n = 0;
  private readonly pending: Promise<void>[] = [];
  constructor(private readonly init: VideoEncoderInit) {}
  static async isConfigSupported(c: VideoEncoderConfig): Promise<VideoEncoderSupport> {
    return { supported: c.codec.startsWith('avc1'), config: c };
  }
  configure(): void {
    this.state = 'configured';
  }
  encode(f: VideoFrame, o?: VideoEncoderEncodeOptions): void {
    const i = this.n++;
    const timestamp = f.timestamp;
    const key = !!o?.keyFrame;
    this.encodeQueueSize++;
    this.pending.push(
      new Promise<void>((r) => setTimeout(r, 1)).then(() => {
        this.encodeQueueSize--;
        if (this.state === 'closed') return;
        const data = Uint8Array.from([0, 0, 0, 2, key ? 0x65 : 0x41, i & 0xff]);
        this.init.output(new EncodedVideoChunk({ type: key ? 'key' : 'delta', timestamp, data }), i === 0 ? { decoderConfig: { codec: 'avc1.640028', description: STAND_IN } } : {});
      }),
    );
  }
  async flush(): Promise<void> {
    await Promise.all(this.pending);
  }
  close(): void {
    this.state = 'closed';
  }
}
(globalThis as unknown as { VideoEncoder: unknown }).VideoEncoder = StubEncoder;
afterAll(() => {
  (globalThis as unknown as { VideoEncoder: unknown }).VideoEncoder = RealEncoder;
});

function sparse(): SparseResult {
  const cameras = Array.from({ length: 12 }, (_, k) => {
    const a = (k / 11) * Math.PI * 0.6;
    const { R, t } = lookAt([1.4 * Math.cos(a), 1.4 * Math.sin(a), 0.6], [0, 0, 0]);
    return { id: k, pose: { R: Float64Array.from(R), t: Float64Array.from(t) }, intrinsics: { width: 480, height: 640, f: 520, cx: 239.5, cy: 319.5, k1: 0, k2: 0 } };
  });
  return {
    points: { count: 0, positions: new Float32Array(0), colors: new Uint8Array(0) },
    cameras,
    metric: { metric: true, source: 'imu', relativeUncertainty: 0.03, gravityAligned: true, northAligned: false, originLatLonAlt: null },
    stats: {},
  };
}

const splat: SplatResult = { count: 50, iterations: 10, ms: 1, shDegree: 0, bytes: 2800, antialiased: true, stats: {} };

function fakeApp() {
  const wakeLock = {
    active: false,
    async acquire(): Promise<void> {
      wakeLock.active = true;
    },
    async release(): Promise<void> {
      wakeLock.active = false;
    },
  };
  const results = { name: 'Vase', sparse: sparse(), splat } as unknown as Results;
  return { results, wakeLock, busy: false, env: { share: false } };
}

/** CPU frames (a grey ramp per frame), rejected once the session is closed. */
function fakeFrames(w: number, h: number): SplatOffscreen & { closed: boolean } {
  const data = new Uint8Array(4 * w * h);
  const s = {
    width: w,
    height: h,
    path: 'readback' as const,
    fallbackReason: null,
    lastFrameMs: 1,
    totalFrameMs: 0,
    frames: 0,
    closed: false,
    async videoFrame(_cam: unknown, timestamp: number, duration?: number) {
      await new Promise((r) => setTimeout(r, 3));
      if (s.closed) throw new DOMException('offscreen session closed', 'AbortError');
      data.fill(40 + (s.frames % 150));
      s.frames++;
      s.totalFrameMs += 1;
      return new VideoFrame(data, { format: 'RGBX', codedWidth: w, codedHeight: h, timestamp, duration });
    },
    async pixels() {
      return { width: w, height: h, data };
    },
    close() {
      s.closed = true;
    },
  };
  return s as unknown as SplatOffscreen & { closed: boolean };
}

function waitFor(pred: () => boolean, timeoutMs = 60_000): Promise<void> {
  return new Promise((resolve, reject) => {
    const t0 = performance.now();
    const tick = (): void => {
      if (pred()) resolve();
      else if (performance.now() - t0 > timeoutMs) reject(new Error('waitFor timed out'));
      else setTimeout(tick, 10);
    };
    tick();
  });
}

function setup() {
  const app = fakeApp();
  const host = document.createElement('div');
  document.body.appendChild(host);
  let building = false;
  let opened: (SplatOffscreen & { closed: boolean }) | null = null;
  const ctl: VideoExportController = createVideoExport({
    app: app as unknown as App,
    openFrames: (o) => (opened = fakeFrames(o.width, o.height)),
    blocked: () => null,
    othersBusy: () => building,
    rerender: () => host.replaceChildren(...[ctl.row()].filter((r): r is HTMLElement => !!r)),
  });
  const q = (id: string): HTMLButtonElement | null => host.querySelector<HTMLButtonElement>(`[data-testid="${id}"]`);
  const tap = async (id: string): Promise<void> => {
    await waitFor(() => !!q(id) && !q(id)!.disabled);
    q(id)!.click();
  };
  return {
    app,
    host,
    ctl,
    q,
    tap,
    setBuilding: (b: boolean) => (building = b),
    frames: () => opened,
    done: () => {
      ctl.dispose();
      host.remove();
    },
  };
}

describe('video export row', () => {
  it('Prepare → progress → Save: an MP4 of the path, busy and wake lock held only meanwhile', async () => {
    const t = setup();
    await waitFor(() => !!t.q('export-prepare-video'));
    expect(t.host.textContent).toMatch(/Fly-through video \(MP4\)/);
    await t.tap('export-prepare-video');
    await waitFor(() => !!t.q('export-cancel-video'));
    expect(t.app.busy).toBe(true);
    expect(t.app.wakeLock.active).toBe(true);
    expect(t.host.textContent).toMatch(/Rendering frame \d+ of 180/);
    // the progress line is updated in place: the row (and its Cancel button) is not rebuilt, so a
    // tap on it cannot fall between two versions of the button
    const cancel = t.q('export-cancel-video');
    const frameNo = (): number => Number(/Rendering frame (\d+) of/.exec(t.host.textContent ?? '')?.[1] ?? -1);
    await waitFor(() => frameNo() >= 100);
    expect(t.q('export-cancel-video')).toBe(cancel);
    await waitFor(() => !!t.q('export-save-video'), 120_000);
    expect(t.app.busy).toBe(false);
    expect(t.app.wakeLock.active).toBe(false);
    expect(t.frames()!.closed).toBe(true);
    const row = t.host.querySelector<HTMLElement>('[data-testid="export-video"]')!;
    expect(row.dataset.codec).toBe('avc1.640028');
    expect(row.textContent).toMatch(/Vase_flythrough\.mp4 · .* · 6\.0 s/);
    let saved: Blob | null = null;
    const create = URL.createObjectURL;
    URL.createObjectURL = (b: Blob | MediaSource) => {
      saved = b as Blob;
      return create.call(URL, b);
    };
    try {
      await t.tap('export-save-video');
      await waitFor(() => !!saved);
    } finally {
      URL.createObjectURL = create;
    }
    const mp4 = readMp4(new Uint8Array(await saved!.arrayBuffer()));
    expect(mp4.samples.length).toBe(180);
    expect([mp4.width, mp4.height]).toEqual([720, 1280]);
    expect(mp4.stss).toEqual([1, 61, 121]);
    t.done();
  });

  it('Cancel, a hidden page, a photoreal build and dispose stop it', async () => {
    const t = setup();
    // Cancel
    await t.tap('export-prepare-video');
    await waitFor(() => (t.frames()?.frames ?? 0) > 5);
    await t.tap('export-cancel-video');
    await waitFor(() => !!t.q('export-prepare-video') && !t.q('export-prepare-video')!.disabled);
    expect(t.frames()!.closed).toBe(true);
    expect([t.app.busy, t.app.wakeLock.active]).toEqual([false, false]);
    // a hidden page
    await t.tap('export-prepare-video');
    await waitFor(() => (t.frames()?.frames ?? 0) > 5);
    let hidden = true;
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => hidden });
    document.dispatchEvent(new Event('visibilitychange'));
    await waitFor(() => !!t.q('export-prepare-video') && !t.q('export-prepare-video')!.disabled);
    expect(t.host.textContent).toMatch(/went to the background/);
    expect([t.app.busy, t.app.wakeLock.active]).toEqual([false, false]);
    hidden = false;
    delete (document as unknown as { hidden?: boolean }).hidden;
    // a photoreal build starts (it holds busy and the wake lock): the export leaves them to it
    await t.tap('export-prepare-video');
    await waitFor(() => (t.frames()?.frames ?? 0) > 5);
    t.setBuilding(true);
    t.app.busy = true;
    await t.app.wakeLock.acquire();
    t.ctl.cancel();
    await waitFor(() => !!t.q('export-prepare-video') && !t.q('export-prepare-video')!.disabled);
    expect([t.app.busy, t.app.wakeLock.active]).toEqual([true, true]);
    t.setBuilding(false);
    t.app.busy = false;
    await t.app.wakeLock.release();
    // dispose mid-export (leaving the viewer): released at once, the late settle changes nothing
    await t.tap('export-prepare-video');
    await waitFor(() => (t.frames()?.frames ?? 0) > 5);
    t.done();
    expect([t.app.busy, t.app.wakeLock.active]).toEqual([false, false]);
    t.app.busy = true; // the next screen's
    await new Promise((r) => setTimeout(r, 100));
    expect(t.app.busy).toBe(true);
    expect(t.frames()!.closed).toBe(true);
  });
});
