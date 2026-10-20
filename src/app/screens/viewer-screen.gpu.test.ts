/**
 * The viewer screen's photoreal build (createViewerScreen with a fake App and a fake engine whose
 * splat() / exportModel() the test settles, in Chromium: the real Viewer, WebGL and — on a
 * hardware adapter, E2E_GPU=host — the WebGPU splat layer): a rebuild drops the Gaussian-splat
 * PLY prepared from the previous model; the previous model's splat layer is freed while the
 * trainer runs and shown again when the build stops; a build settling after the screen was left
 * does not take the wake lock / busy flag of the next screen.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { getGPU } from '../../gpu/device';
import { lookAt } from '../../splat/testutil';
import { SH_C0, allocSplatModel, logit } from '../../splat/types';
import type { Viewer } from '../../viewer';
import { ENGINE_ERRORS, namedError, type ExportFormat, type SparseResult, type SplatModelPayload, type SplatOptions, type SplatResult } from '../../workers/protocol';
import type { App, Results } from '../app';
import { closeAllModals } from '../components/layers';
import { createViewerScreen } from './viewer';

class FakeEngine {
  readonly builds: { opts?: SplatOptions; resolve: (r: SplatResult) => void; reject: (e: Error) => void }[] = [];
  readonly aborts: unknown[] = [];
  /** The stored model (splatModel / exportModel read it). */
  model: SplatModelPayload | null = null;
  modelLoads = 0;
  /** Exports wait for this while set. */
  exportGate: Promise<void> | null = null;
  splat(opts?: SplatOptions): Promise<SplatResult> {
    return new Promise((resolve, reject) => this.builds.push({ opts, resolve, reject }));
  }
  async splatModel(): Promise<SplatModelPayload | null> {
    this.modelLoads++;
    return this.model;
  }
  async exportModel(format: ExportFormat): Promise<Blob> {
    const text = `${format}:${this.model?.model.count ?? 0}`;
    if (this.exportGate) await this.exportGate;
    return new Blob([text]);
  }
  async abortLongOperation(opts?: unknown): Promise<'cancelled'> {
    this.aborts.push(opts);
    return 'cancelled';
  }
}

function sparse(): SparseResult {
  const n = 300;
  const positions = new Float32Array(3 * n);
  for (let i = 0; i < 3 * n; i++) positions[i] = Math.sin(1.7 * i + 0.3) * 0.5;
  const camera = (eye: [number, number, number]) => {
    const { R, t } = lookAt(eye, [0, 0, 0]);
    return { R: Float64Array.from(R), t: Float64Array.from(t) };
  };
  const intrinsics = { width: 640, height: 480, f: 500, cx: 319.5, cy: 239.5, k1: 0, k2: 0 };
  return {
    points: { count: n, positions, colors: new Uint8Array(3 * n).fill(160) },
    cameras: [
      { id: 0, pose: camera([0.4, -2, 0.5]), intrinsics },
      { id: 1, pose: camera([-0.4, -2, 0.5]), intrinsics },
    ],
    metric: { metric: false, source: 'none', relativeUncertainty: null, gravityAligned: false, northAligned: false, originLatLonAlt: null },
    stats: {},
  };
}

function payload(count: number): SplatModelPayload {
  const m = allocSplatModel(count, 0);
  for (let i = 0; i < count; i++) {
    for (let k = 0; k < 3; k++) m.means[3 * i + k] = Math.sin(2.3 * i + k) * 0.4;
    m.logScales.fill(Math.log(0.03), 3 * i, 3 * i + 3);
    m.opacities[i] = logit(0.9);
    for (let c = 0; c < 3; c++) m.sh[3 * i + c] = (0.3 + 0.2 * c - 0.5) / SH_C0;
  }
  return { model: m, antialiased: true, background: [0, 0, 0], iterations: 100 };
}

function info(count: number): SplatResult {
  return { count, iterations: 100, ms: 1, shDegree: 0, bytes: 56 * count, antialiased: true, stats: {} };
}

function fakeApp(engine: FakeEngine, splat: SplatResult | null) {
  const wakeLock = {
    active: false,
    async acquire(): Promise<void> {
      wakeLock.active = true;
    },
    async release(): Promise<void> {
      wakeLock.active = false;
    },
  };
  const results: Results = { sessionId: 's1', name: 'Bench', mode: 'object', quality: 'fast', sparse: sparse(), dense: null, mesh: null, splat, liveSparse: null, liveCameras: null, stats: {} };
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
  for (const s of mounted) leave(s);
});

function mount(app: ReturnType<typeof fakeApp>): { el: HTMLElement; leave(): void } {
  const screen = createViewerScreen(app as unknown as App, { screen: 'viewer', sessionId: 's1' });
  screen.el.style.cssText = 'position:fixed;left:0;top:0;width:320px;height:480px';
  document.body.appendChild(screen.el);
  screen.enter?.();
  mounted.add(screen);
  return screen;
}

function leave(s: { el: HTMLElement; leave(): void }): void {
  if (!mounted.delete(s)) return;
  s.leave();
  s.el.remove();
}

function waitFor(pred: () => boolean, timeoutMs = 20_000): Promise<void> {
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

/** Close the open sheet and wait until it is gone (its rows must not answer the next query). */
async function closeSheets(): Promise<void> {
  closeAllModals();
  await waitFor(() => !document.querySelector('.sheet-wrap'));
}

/** Viewer screen loaded, its splat model (if any) set or refused. */
async function loaded(app: ReturnType<typeof fakeApp>, screen: { el: HTMLElement }): Promise<void> {
  await waitFor(() => !!app.debug.viewer);
  if (app.results.splat) await waitFor(() => ['ready', 'unavailable'].includes(screen.el.dataset.splats ?? ''));
}

/** Build sheet → quality → the photoreal item's Build / Rebuild / Resume. */
async function startBuild(quality: 'fast' | 'balanced' | 'max'): Promise<void> {
  await tap('tool-build');
  await waitFor(() => !!q('sheet-build'));
  document.querySelector<HTMLElement>(`[data-testid="sheet-build"] .segment[data-value="${quality}"]`)!.click();
  await tap('build-splat');
  await closeSheets();
}

describe('viewer screen: photoreal build', () => {
  it('a rebuild drops the Gaussian-splat PLY prepared from the previous model (review)', async () => {
    const engine = new FakeEngine();
    engine.model = payload(40);
    const app = fakeApp(engine, info(40));
    const screen = mount(app);
    await loaded(app, screen);
    // the splat PLY of the current model is prepared
    await tap('tool-export');
    await tap('export-prepare-ply-splat');
    await waitFor(() => !!q('export-save-ply-splat'));
    await closeSheets();
    // rebuild at another quality
    await startBuild('balanced');
    expect(engine.builds.length).toBe(1);
    engine.model = payload(60);
    engine.builds[0].resolve(info(60));
    await waitFor(() => q('splat-progress')!.hidden === true);
    await tap('tool-export');
    await waitFor(() => !!q('export-ply-splat'));
    expect(q('export-save-ply-splat')).toBeNull();
    expect(q('export-prepare-ply-splat')).not.toBeNull();
    await closeSheets();

    // an export prepared DURING a rebuild and finishing after it may hold the previous model
    let open!: () => void;
    engine.exportGate = new Promise((r) => (open = r));
    await startBuild('max');
    await tap('tool-export');
    await tap('export-prepare-ply-splat');
    await closeSheets();
    engine.model = payload(80);
    engine.builds[1].resolve(info(80));
    await waitFor(() => q('splat-progress')!.hidden === true);
    open();
    await new Promise((r) => setTimeout(r, 30));
    await tap('tool-export');
    await waitFor(() => !!q('export-ply-splat'));
    expect(q('export-save-ply-splat')).toBeNull();
    expect(q('export-prepare-ply-splat')).not.toBeNull();
  });

  it('a build that settles after the screen was left keeps the next screen’s wake lock and busy flag (review)', async () => {
    const engine = new FakeEngine();
    const app = fakeApp(engine, null);
    // an interrupted densify: its Resume in the Build sheet waits while the photoreal model trains
    app.results.stats = { denseResumablePass: 1 };
    const screen = mount(app);
    await loaded(app, screen);
    await startBuild('fast');
    expect(engine.builds.length).toBe(1);
    expect(app.wakeLock.active).toBe(true);
    expect(app.busy).toBe(true);
    await tap('tool-build');
    await waitFor(() => !!q('build-dense-resume'));
    expect((q('build-dense') as HTMLButtonElement).disabled).toBe(true);
    await closeSheets();
    // the user leaves: the build is aborted, the next screen takes the lock and the busy flag at once
    leave(screen);
    expect(engine.aborts).toEqual([{ graceMs: 8000, method: 'splat' }]);
    expect(app.wakeLock.active).toBe(false);
    expect(app.busy).toBe(false);
    await app.wakeLock.acquire();
    app.busy = true;
    // the aborted build settles later
    engine.builds[0].reject(namedError(ENGINE_ERRORS.cancelled, 'Cancelled'));
    await new Promise((r) => setTimeout(r, 30));
    expect(app.wakeLock.active).toBe(true);
    expect(app.busy).toBe(true);
  });

  it('a rebuild frees the shown splat layer while the trainer runs and shows the stored model again when it stops (review)', async () => {
    const gpu = await getGPU();
    // the screen's viewer refuses software adapters: the layer exists on a hardware GPU only
    const hw = !!gpu && !gpu.isSoftware;
    const engine = new FakeEngine();
    engine.model = payload(40);
    const app = fakeApp(engine, info(40));
    const screen = mount(app);
    await loaded(app, screen);
    const viewer = app.debug.viewer as Viewer;
    if (hw) {
      expect(screen.el.dataset.splats).toBe('ready');
      expect(viewer.getStats().splatLayer).not.toBeNull();
      expect(q('layer-switch')!.dataset.layer).toBe('splats');
    }
    const loads = engine.modelLoads;
    await startBuild('balanced');
    expect(engine.builds.length).toBe(1);
    // the old layer is not in the engine's page reserve: freed for the training, points shown
    expect(viewer.getStats().splatLayer).toBeNull();
    expect(viewer.getStats().splatsShown).toBe(false);
    expect(document.querySelector('[data-testid="layer-switch"] .segment[data-value="splats"]')).toBeNull();
    // cancelled: the previous model (still the stored one) comes back
    engine.builds[0].reject(namedError(ENGINE_ERRORS.cancelled, 'Cancelled'));
    await waitFor(() => q('splat-progress')!.hidden === true);
    if (hw) {
      await waitFor(() => screen.el.dataset.splats === 'ready' && viewer.getStats().splatLayer !== null);
      expect(engine.modelLoads).toBe(loads + 1);
      await waitFor(() => q('layer-switch')!.dataset.layer === 'splats');
    }
    // a successful rebuild shows the new model
    await startBuild('balanced');
    expect(viewer.getStats().splatLayer).toBeNull();
    engine.model = payload(70);
    engine.builds[1].resolve(info(70));
    if (hw) await waitFor(() => screen.el.dataset.splats === 'ready' && viewer.getStats().splatLayer?.gaussians === 70);
  });
});
