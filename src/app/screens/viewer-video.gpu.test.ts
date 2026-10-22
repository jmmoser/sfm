/**
 * The viewer screen's "Fly-through video (MP4)" export row (src/app/video-export.ts) with a fake
 * App / engine in Chromium: shown only when a splat model exists AND VideoEncoder supports H.264
 * (isConfigSupported), unavailable while the splat layer cannot draw (SwiftShader: the screen's
 * viewer refuses software adapters). On a hardware GPU with an H.264 encoder (E2E_GPU=host, Chrome)
 * the whole flow: Prepare → progress → Cancel; Prepare → Save → a decodable MP4; a hidden page
 * stops the export. (Own file: the encoder support is cached per page.)
 */
import { afterEach, describe, expect, it } from 'vitest';
import { getGPU } from '../../gpu/device';
import { readMp4 } from '../../io/mp4';
import { lookAt } from '../../splat/testutil';
import { SH_C0, allocSplatModel, logit } from '../../splat/types';
import type { SparseResult, SplatModelPayload, SplatResult } from '../../workers/protocol';
import type { App, Results } from '../app';
import { closeAllModals } from '../components/layers';
import { chooseVideoConfig } from '../video-export';
import { createViewerScreen } from './viewer';

function sparse(): SparseResult {
  const n = 400;
  const positions = new Float32Array(3 * n);
  for (let i = 0; i < 3 * n; i++) positions[i] = Math.sin(1.7 * i + 0.3) * 0.4;
  // a half orbit of 12 photos around the origin
  const cameras = Array.from({ length: 12 }, (_, k) => {
    const a = (k / 11) * Math.PI;
    const { R, t } = lookAt([1.4 * Math.cos(a), 1.4 * Math.sin(a), 0.6], [0, 0, 0]);
    return { id: k, pose: { R: Float64Array.from(R), t: Float64Array.from(t) }, intrinsics: { width: 480, height: 640, f: 520, cx: 239.5, cy: 319.5, k1: 0, k2: 0 } };
  });
  return {
    points: { count: n, positions, colors: new Uint8Array(3 * n).fill(160) },
    cameras,
    metric: { metric: true, source: 'imu', relativeUncertainty: 0.03, gravityAligned: true, northAligned: false, originLatLonAlt: null },
    stats: {},
  };
}

function payload(count: number): SplatModelPayload {
  const m = allocSplatModel(count, 0);
  for (let i = 0; i < count; i++) {
    for (let k = 0; k < 3; k++) m.means[3 * i + k] = Math.sin(2.3 * i + k) * 0.35;
    m.logScales.fill(Math.log(0.04), 3 * i, 3 * i + 3);
    m.opacities[i] = logit(0.9);
    for (let c = 0; c < 3; c++) m.sh[3 * i + c] = (0.3 + 0.2 * c - 0.5) / SH_C0;
  }
  return { model: m, antialiased: true, background: [0, 0, 0], iterations: 100 };
}

const info = (count: number): SplatResult => ({ count, iterations: 100, ms: 1, shDegree: 0, bytes: 56 * count, antialiased: true, stats: {} });

function fakeApp(splat: SplatResult | null) {
  const engine = {
    model: splat ? payload(splat.count) : null,
    async splatModel(): Promise<SplatModelPayload | null> {
      return engine.model;
    },
    async exportModel(): Promise<Blob> {
      return new Blob(['x']);
    },
    async abortLongOperation(): Promise<'cancelled'> {
      return 'cancelled';
    },
  };
  const wakeLock = {
    active: false,
    async acquire(): Promise<void> {
      wakeLock.active = true;
    },
    async release(): Promise<void> {
      wakeLock.active = false;
    },
  };
  const results: Results = { sessionId: 's1', name: 'Vase', mode: 'object', quality: 'fast', sparse: sparse(), dense: null, mesh: null, splat, liveSparse: null, liveCameras: null, stats: {} };
  return {
    results,
    engine,
    wakeLock,
    busy: false,
    loadedSessionId: 's1' as string | null,
    env: { webgpu: 'available', webgl: true, share: false },
    debug: {} as { viewer?: unknown },
    navigate: (): void => undefined,
    updateSession: async (): Promise<void> => undefined,
    ensureSessionLoaded: async (): Promise<boolean> => true,
  };
}

const mounted = new Set<{ el: HTMLElement; leave(): void }>();
afterEach(() => {
  closeAllModals();
  for (const s of mounted) {
    s.leave();
    s.el.remove();
  }
  mounted.clear();
});

function mount(app: ReturnType<typeof fakeApp>): { el: HTMLElement; leave(): void } {
  const screen = createViewerScreen(app as unknown as App, { screen: 'viewer', sessionId: 's1' });
  screen.el.style.cssText = 'position:fixed;left:0;top:0;width:320px;height:480px';
  document.body.appendChild(screen.el);
  screen.enter?.();
  mounted.add(screen);
  return screen;
}

function waitFor(pred: () => boolean, timeoutMs = 60_000): Promise<void> {
  return new Promise((resolve, reject) => {
    const t0 = performance.now();
    const tick = (): void => {
      if (pred()) resolve();
      else if (performance.now() - t0 > timeoutMs) reject(new Error('waitFor timed out'));
      else setTimeout(tick, 20);
    };
    tick();
  });
}

const q = (id: string): HTMLElement | null => document.querySelector<HTMLElement>(`[data-testid="${id}"]`);

async function tap(id: string): Promise<void> {
  await waitFor(() => !!q(id) && !(q(id) as HTMLButtonElement).disabled);
  q(id)!.click();
}

async function openExportSheet(app: ReturnType<typeof fakeApp>, screen: { el: HTMLElement }): Promise<void> {
  await waitFor(() => !!app.debug.viewer);
  if (app.results.splat) await waitFor(() => ['ready', 'unavailable'].includes(screen.el.dataset.splats ?? ''));
  await tap('tool-export');
  await waitFor(() => !!q('sheet-export'));
  // the encoder support probe settles (cached) and re-renders the sheet
  await new Promise((r) => setTimeout(r, 100));
}

describe('viewer screen: fly-through video export row', () => {
  it('is shown only with a splat model and an H.264 encoder; unavailable while the splats cannot draw', async () => {
    const gpu = await getGPU();
    const hw = !!gpu && !gpu.isSoftware;
    const h264 = !!(await chooseVideoConfig(720, 1280));
    if (!h264) {
      // pretend H.264 is supported (isConfigSupported only: nothing is encoded here)
      const orig = VideoEncoder.isConfigSupported.bind(VideoEncoder);
      VideoEncoder.isConfigSupported = async (c: VideoEncoderConfig) => (c.codec.startsWith('avc1') ? { supported: true, config: c } : orig(c));
    }
    // no splat model: no row
    const a = fakeApp(null);
    const s1 = mount(a);
    await openExportSheet(a, s1);
    expect(q('export-video')).toBeNull();
    closeAllModals();
    s1.leave();
    mounted.delete(s1);
    s1.el.remove();
    // a splat model: the row
    const b = fakeApp(info(80));
    const s2 = mount(b);
    await openExportSheet(b, s2);
    await waitFor(() => !!q('export-video'));
    if (!hw) {
      // SwiftShader: the screen's viewer has no splat layer → unavailable, no Prepare
      expect(s2.el.dataset.splats).toBe('unavailable');
      expect(q('export-video')!.classList.contains('disabled')).toBe(true);
      expect(q('export-video')!.textContent).toMatch(/cannot be shown/);
      expect(q('export-prepare-video')).toBeNull();
    } else expect(q('export-prepare-video')).not.toBeNull();
  });

  it('hardware GPU + H.264: Prepare → progress → Cancel; Prepare → Save → MP4; hidden page stops it', async () => {
    const gpu = await getGPU();
    if (!gpu || gpu.isSoftware || !(await chooseVideoConfig(720, 1280))) {
      console.warn('[viewer-video] needs a hardware GPU and an H.264 encoder (E2E_GPU=host): skipped');
      return;
    }
    const app = fakeApp(info(120));
    const screen = mount(app);
    await openExportSheet(app, screen);
    await tap('export-prepare-video');
    await waitFor(() => !!q('export-cancel-video'));
    expect(app.wakeLock.active).toBe(true);
    await tap('export-cancel-video');
    await waitFor(() => !!q('export-prepare-video'));
    expect(app.wakeLock.active).toBe(false);
    // hidden page: the export stops
    await tap('export-prepare-video');
    await waitFor(() => !!q('export-cancel-video'));
    let hidden = true;
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => hidden });
    document.dispatchEvent(new Event('visibilitychange'));
    await waitFor(() => !!q('export-prepare-video'));
    expect(q('export-video')!.textContent).toMatch(/background/);
    hidden = false;
    delete (document as unknown as { hidden?: boolean }).hidden;
    // a full export, then Save (download path: the Blob URL is captured)
    await tap('export-prepare-video');
    await waitFor(() => !!q('export-save-video'), 300_000);
    const row = q('export-video')!;
    console.log(`[viewer-video] ${row.textContent} · ${row.dataset.codec} · ${row.dataset.framePath} · ${row.dataset.msPerFrame} ms/frame`);
    let saved: Blob | null = null;
    const create = URL.createObjectURL;
    URL.createObjectURL = (b: Blob | MediaSource) => {
      saved = b as Blob;
      return create.call(URL, b);
    };
    try {
      await tap('export-save-video');
      await waitFor(() => !!saved);
    } finally {
      URL.createObjectURL = create;
    }
    const mp4 = readMp4(new Uint8Array(await saved!.arrayBuffer()));
    expect(mp4.samples.length).toBeGreaterThanOrEqual(180);
    expect(mp4.samples.length).toBeLessThanOrEqual(300);
  });
});
