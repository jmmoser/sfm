/**
 * The Processing screen and the run journal (createProcessingScreen with a fake App and a fake
 * engine whose refine() / densify() / mesh() the test settles, in Chromium: the real preview
 * Viewer): each step is a journal run, closed on success, on Stop and when the screen is left (the
 * 8 s Stop fallback) — so the next launch reports nothing; after a kill of a step, that step runs
 * with a light preview (at most LIGHT_MODE_MAX_POINTS of the dense cloud) and the next step, not
 * killed, shows the full cloud again.
 */
import { afterEach, describe, expect, it } from 'vitest';
import type { PointCloud, ProgressEvent, TriangleMesh } from '../../core/types';
import { lookAt } from '../../splat/testutil';
import { ENGINE_ERRORS, namedError, type SparseResult } from '../../workers/protocol';
import type { App, Results } from '../app';
import { closeAllModals } from '../components/layers';
import type { NavEvent, ProcessingStep } from '../router';
import { LIGHT_MODE_MAX_POINTS, RUN_JOURNAL_KEY, RunJournal, parseRunRecord, type JournalStorage, type RunStep } from '../run-journal';
import { createProcessingScreen } from './processing';

interface Call {
  step: ProcessingStep;
  progress?: (e: ProgressEvent) => void;
  resolve: (r: unknown) => void;
  reject: (e: Error) => void;
}

class FakeEngine {
  readonly calls: Call[] = [];
  cancels = 0;
  status = { state: 'ready' };
  private call<T>(step: ProcessingStep, progress?: (e: ProgressEvent) => void): Promise<T> {
    return new Promise<T>((resolve, reject) => this.calls.push({ step, progress, resolve: resolve as (r: unknown) => void, reject }));
  }
  refine(progress?: (e: ProgressEvent) => void): Promise<SparseResult> {
    return this.call('refine', progress);
  }
  densify(_opts: unknown, progress?: (e: ProgressEvent) => void): Promise<PointCloud> {
    return this.call('densify', progress);
  }
  mesh(_opts: unknown, progress?: (e: ProgressEvent) => void): Promise<TriangleMesh> {
    return this.call('mesh', progress);
  }
  cancel(): void {
    this.cancels++;
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

function cloud(n: number): PointCloud {
  const positions = new Float32Array(3 * n);
  for (let i = 0; i < 3 * n; i++) positions[i] = Math.sin(1.3 * i + 0.7) * 0.5;
  return { count: n, positions, colors: new Uint8Array(3 * n).fill(200), radii: new Float32Array(n).fill(0.004) };
}

function quad(): TriangleMesh {
  return {
    vertexCount: 4,
    triangleCount: 2,
    positions: new Float32Array([-0.3, -0.3, 0, 0.3, -0.3, 0, 0.3, 0.3, 0, -0.3, 0.3, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
    colors: new Uint8Array(12).fill(150),
    indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  };
}

function memStorage(): JournalStorage & { map: Map<string, string> } {
  const map = new Map<string, string>();
  return { map, getItem: (k) => map.get(k) ?? null, setItem: (k, v) => void map.set(k, v), removeItem: (k) => void map.delete(k) };
}

/** The journal at the launch after a run of `step` of scan s1 was killed (the card answered). */
function killed(storage: JournalStorage, step: RunStep): RunJournal {
  new RunJournal(storage).begin({ sessionId: 's1', name: 'Wall', step, stage: 'depth_maps', quality: 'fast' });
  const j = new RunJournal(storage);
  j.recover();
  j.acknowledge();
  return j;
}

function fakeApp(engine: FakeEngine, journal: RunJournal) {
  const results: Results = { sessionId: 's1', name: 'Wall', mode: 'scene', quality: 'fast', sparse: sparse(), dense: null, mesh: null, splat: null, liveSparse: null, liveCameras: null, stats: {} };
  const nav: NavEvent[] = [];
  return {
    results,
    engine,
    journal,
    nav,
    loadedSessionId: 's1' as string | null,
    env: { webgl: true },
    wakeLock: { acquire: async (): Promise<void> => undefined, release: async (): Promise<void> => undefined },
    navigate: (e: NavEvent): void => void nav.push(e),
    updateSession: async (): Promise<void> => undefined,
    ensureSessionLoaded: async (): Promise<boolean> => true,
  };
}

const mounted = new Set<{ el: HTMLElement; leave(): void }>();
afterEach(() => {
  closeAllModals();
  for (const s of mounted) leave(s);
});

function mount(app: ReturnType<typeof fakeApp>, steps: ProcessingStep[]): { el: HTMLElement; leave(): void } {
  const screen = createProcessingScreen(app as unknown as App, { screen: 'processing', sessionId: 's1', steps });
  screen.el.style.cssText = 'position:fixed;left:0;top:0;width:320px;height:560px';
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
const caption = (): string => document.querySelector('.proc-preview-caption')?.textContent ?? '';
const record = (storage: { map: Map<string, string> }) => parseRunRecord(storage.map.get(RUN_JOURNAL_KEY) ?? null);

describe('processing screen: run journal', () => {
  it('after a kill of a step, that step runs with a light preview; the next step shows the full cloud again (review)', async () => {
    const storage = memStorage();
    const engine = new FakeEngine();
    const app = fakeApp(engine, killed(storage, 'densify'));
    // a dense rebuild: the previous cloud, larger than the light-mode cap, is in the preview
    app.results.dense = cloud(LIGHT_MODE_MAX_POINTS + 50_000);
    mount(app, ['densify', 'mesh']);
    await waitFor(() => engine.calls.length === 1 && /shown\)/.test(caption()));
    expect(q('processing-preview')!.dataset.light).toBe('1');
    expect(caption()).toBe('Dense cloud · 1.55M points (1.5M shown)');
    expect(record(storage)).toMatchObject({ step: 'densify', kills: { densify: 1 } });

    // the densify completes: its new cloud is shown light too, then the mesh (not killed) runs with the full one
    engine.calls[0].resolve(cloud(LIGHT_MODE_MAX_POINTS + 80_000));
    await waitFor(() => engine.calls.length === 2);
    expect(engine.calls[1].step).toBe('mesh');
    expect(q('processing-preview')!.dataset.light).toBeUndefined();
    expect(caption()).toBe('Dense cloud · 1.58M points');
    expect(record(storage)).toMatchObject({ step: 'mesh', kills: {} });

    engine.calls[1].resolve(quad());
    await waitFor(() => app.nav.length === 1);
    expect(app.nav[0]).toEqual({ type: 'PROCESSING_DONE' });
    expect(record(storage)).toMatchObject({ step: null, kills: {} });
    expect(new RunJournal(storage).recover()).toBeNull();
  });

  it('Stop closes the run (no card at the next launch) and keeps the kill count (review)', async () => {
    const storage = memStorage();
    const engine = new FakeEngine();
    const app = fakeApp(engine, killed(storage, 'densify'));
    mount(app, ['densify']);
    await waitFor(() => engine.calls.length === 1);
    // (a sparse scan: the preview is the sparse model, light mode changes nothing there)
    expect(caption()).toMatch(/^Sparse model/);
    engine.calls[0].progress?.({ stage: 'fusion', progress: 0.2 });
    expect(record(storage)).toMatchObject({ step: 'densify', stage: 'fusion' });
    q('processing-cancel')!.click();
    await waitFor(() => !!q('dialog-confirm'));
    q('dialog-confirm')!.click();
    await waitFor(() => engine.cancels === 1);
    engine.calls[0].reject(namedError(ENGINE_ERRORS.cancelled, 'Cancelled'));
    await waitFor(() => app.nav.length === 1);
    expect(app.nav[0]).toEqual({ type: 'PROCESSING_CANCELLED', hasModel: true });
    expect(record(storage)).toMatchObject({ step: null, kills: { densify: 1 } });
    expect(new RunJournal(storage).recover()).toBeNull();
  });

  it('left while a step runs (the 8 s Stop fallback: the engine did not stop) is no interruption (review)', async () => {
    const storage = memStorage();
    const engine = new FakeEngine();
    const app = fakeApp(engine, new RunJournal(storage));
    const screen = mount(app, ['refine']);
    await waitFor(() => engine.calls.length === 1);
    expect(record(storage)).toMatchObject({ step: 'refine', stage: 'matching' });
    leave(screen);
    expect(record(storage)).toMatchObject({ step: null, kills: {} });
    // the engine stops later: nothing more is written
    engine.calls[0].reject(namedError(ENGINE_ERRORS.cancelled, 'Cancelled'));
    await new Promise((r) => setTimeout(r, 30));
    expect(record(storage)).toMatchObject({ step: null, kills: {} });
    expect(new RunJournal(storage).recover()).toBeNull();
  });
});
