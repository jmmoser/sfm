/**
 * "View in AR" (src/app/ar-quicklook.ts) in Chromium: availability (Apple mobile WebKit with
 * rel="ar" support only), the launch anchor (<a rel="ar" download><img></a> on a blob URL that is
 * reused for the same Blob and revoked when replaced / disposed), the Export sheet's USDZ row
 * (createViewerScreen with a fake engine; no WebGL viewer), and the MockEngine's browser USDZ
 * (JPEG atlas through OffscreenCanvas, 64-byte aligned package).
 */
import { afterEach, describe, expect, it } from 'vitest';
import type { TriangleMesh } from '../core/types';
import { marchingCubes } from '../dense/marching-cubes';
import { TsdfVolume } from '../dense/tsdf';
import { readZip } from '../io/zip';
import type { ExportFormat, SparseResult } from '../workers/protocol';
import type { App, Results } from './app';
import { arQuickLookAvailable, createARQuickLook } from './ar-quicklook';
import { closeAllModals } from './components/layers';
import { MockEngine } from './mock-engine';
import { createViewerScreen } from './screens/viewer';

/** Pretend this browser supports rel="ar" (Chromium does not). */
function stubRelAr(on: boolean): () => void {
  const proto = DOMTokenList.prototype;
  const orig = proto.supports;
  proto.supports = function (this: DOMTokenList, token: string): boolean {
    return on && token === 'ar' ? true : orig.call(this, token);
  };
  return () => {
    proto.supports = orig;
  };
}

/** Captures (and cancels) clicks on rel="ar" anchors. */
function captureArClicks(): { anchors: HTMLAnchorElement[]; stop(): void } {
  const anchors: HTMLAnchorElement[] = [];
  const onClick = (ev: MouseEvent): void => {
    const a = ev.target as HTMLAnchorElement;
    if (a instanceof HTMLAnchorElement && a.rel === 'ar') {
      ev.preventDefault();
      anchors.push(a);
    }
  };
  document.addEventListener('click', onClick, true);
  return { anchors, stop: () => document.removeEventListener('click', onClick, true) };
}

const fetchOk = async (url: string): Promise<boolean> => {
  try {
    await (await fetch(url)).arrayBuffer();
    return true;
  } catch {
    return false;
  }
};

const cleanups: (() => void)[] = [];
afterEach(async () => {
  closeAllModals();
  while (cleanups.length) cleanups.pop()!();
  // (a closing sheet stays in the DOM while it animates out: its rows must not answer the next test)
  const t0 = performance.now();
  while (document.querySelector('.sheet-wrap') && performance.now() - t0 < 5000) await new Promise((r) => setTimeout(r, 20));
});

describe('AR Quick Look link', () => {
  it('is available only on Apple mobile WebKit with rel="ar" support', () => {
    expect(arQuickLookAvailable({ apple: true })).toBe(false); // Chromium: no rel="ar"
    cleanups.push(stubRelAr(true));
    expect(arQuickLookAvailable({ apple: true })).toBe(true);
    // relList.supports('ar') alone is not trusted (macOS Safari, in-app browsers: WebKit 239135)
    expect(arQuickLookAvailable({ apple: false })).toBe(false);
  });

  it('clicks a hidden <a rel="ar" download><img></a> on a reused blob URL; revokes replaced and disposed URLs', async () => {
    const cap = captureArClicks();
    cleanups.push(cap.stop);
    const ql = createARQuickLook();
    const a = new Blob([new Uint8Array([1, 2, 3])], { type: 'model/vnd.usdz+zip' });
    ql.open(a, 'My scan: ar.usdz');
    ql.open(a, 'My scan: ar.usdz');
    expect(cap.anchors.length).toBe(2);
    const [x, y] = cap.anchors;
    expect(x.href).toMatch(/^blob:/);
    expect(y.href).toBe(x.href);
    expect(x.download).toBe('My scan_ ar.usdz');
    expect(x.firstElementChild?.tagName).toBe('IMG');
    expect(x.style.display).toBe('none');
    expect(await fetchOk(x.href)).toBe(true);
    // a newer export replaces the URL
    ql.open(new Blob([new Uint8Array([4])], { type: 'model/vnd.usdz+zip' }), 'b.usdz');
    const z = cap.anchors[2];
    expect(z.href).not.toBe(x.href);
    expect(await fetchOk(x.href)).toBe(false);
    ql.dispose();
    expect(await fetchOk(z.href)).toBe(false);
  });
});

describe('Export sheet: USDZ row', () => {
  function sparse(): SparseResult {
    return {
      points: { count: 1, positions: new Float32Array(3), colors: new Uint8Array(3) },
      cameras: [],
      metric: { metric: true, source: 'imu', relativeUncertainty: 0.01, gravityAligned: true, northAligned: false, originLatLonAlt: null },
      stats: {},
    };
  }
  const mesh: TriangleMesh = { vertexCount: 3, triangleCount: 1, positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), normals: new Float32Array(9), colors: new Uint8Array(9), indices: new Uint32Array([0, 1, 2]) };

  function mountViewer(apple: boolean): { exports: ExportFormat[] } {
    const exports: ExportFormat[] = [];
    const engine = {
      async exportModel(format: ExportFormat): Promise<Blob> {
        exports.push(format);
        return new Blob([format], { type: format === 'usdz' ? 'model/vnd.usdz+zip' : 'application/octet-stream' });
      },
    };
    const results: Results = { sessionId: 's1', name: 'Mug', mode: 'object', quality: 'fast', sparse: sparse(), dense: null, mesh, splat: null, liveSparse: null, liveCameras: null, stats: {} };
    const app = {
      results,
      engine,
      busy: false,
      loadedSessionId: 's1',
      env: { webgpu: 'available', webgl: false, share: false, apple },
      debug: {},
      navigate: (): void => undefined,
      ensureSessionLoaded: async (): Promise<boolean> => true,
    };
    const screen = createViewerScreen(app as unknown as App, { screen: 'viewer', sessionId: 's1' });
    document.body.appendChild(screen.el);
    screen.enter?.();
    cleanups.push(() => {
      screen.leave();
      screen.el.remove();
    });
    return { exports };
  }

  const q = (id: string): HTMLElement | null => document.querySelector<HTMLElement>(`[data-testid="${id}"]`);
  async function waitFor(pred: () => boolean): Promise<void> {
    const t0 = performance.now();
    while (!pred()) {
      if (performance.now() - t0 > 10_000) throw new Error('waitFor timed out');
      await new Promise((r) => setTimeout(r, 20));
    }
  }
  async function prepareUsdz(): Promise<void> {
    await waitFor(() => !!q('tool-export'));
    q('tool-export')!.click();
    await waitFor(() => !!q('export-prepare-usdz'));
    expect(q('export-usdz')!.textContent).toContain('true size');
    q('export-prepare-usdz')!.click();
    await waitFor(() => !!q('export-save-usdz'));
  }

  it('offers "View in AR" next to Save on iPhone / iPad Safari and opens AR Quick Look on the prepared file', async () => {
    cleanups.push(stubRelAr(true));
    const cap = captureArClicks();
    cleanups.push(cap.stop);
    const { exports } = mountViewer(true);
    await prepareUsdz();
    expect(exports).toEqual(['usdz']);
    const view = q('export-ar-usdz')!;
    expect(view.textContent).toContain('View in AR');
    // one primary action per row
    expect(q('export-save-usdz')!.classList.contains('btn-secondary')).toBe(true);
    view.click();
    expect(cap.anchors.length).toBe(1);
    expect(cap.anchors[0].download).toBe('Mug_ar.usdz');
    expect(await (await fetch(cap.anchors[0].href)).text()).toBe('usdz');
  });

  it('has no "View in AR" elsewhere (Share / Save only)', async () => {
    cleanups.push(stubRelAr(true));
    mountViewer(false);
    await prepareUsdz();
    expect(q('export-ar-usdz')).toBeNull();
    expect(q('export-save-usdz')!.classList.contains('btn-primary')).toBe(true);
  });
});

describe('MockEngine USDZ (browser)', () => {
  it('bakes a JPEG atlas into a 64-byte aligned package', async () => {
    const vol = TsdfVolume.fromFunction((x, y, z) => Math.hypot(x, y, z - 0.2) - 0.15, [-0.2, -0.2, 0], [0.2, 0.2, 0.4], 0.02);
    const m = marchingCubes(vol);
    m.colors.fill(200);
    const e = new MockEngine({ speed: 50, sessions: async () => null });
    (e as unknown as { meshModel: TriangleMesh }).meshModel = m;
    const zip = new Uint8Array(await (await e.exportModel('usdz')).arrayBuffer());
    const entries = readZip(zip);
    expect(entries.map((x) => x.name)).toEqual(['model.usda', 'textures/albedo.jpg']);
    for (const x of entries) expect((x.data.byteOffset - zip.byteOffset) % 64).toBe(0);
    expect([entries[1].data[0], entries[1].data[1]]).toEqual([0xff, 0xd8]); // JPEG SOI
    expect(new TextDecoder().decode(entries[0].data)).toContain('upAxis = "Y"');
    e.dispose();
  });
});
