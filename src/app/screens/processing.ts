/**
 * Processing: runs the scheduled steps (refine → densify → mesh) on the engine and shows the
 * six stages (Matching, Reconstruction, Alignment, Depth maps, Fusion, Meshing) with per-stage
 * progress bars, elapsed time and an ETA, a live 3D preview of the sparse model, and Cancel.
 * Continues to the Viewer when done. Holds the wake lock (iOS suspends a hidden page).
 */
import type { PointCloud, ProgressEvent } from '../../core/types';
import type { Viewer } from '../../viewer';
import { loadViewerModule } from '../viewer-loader';
import { ENGINE_ERRORS, isCancelledError } from '../../workers/protocol';
import type { App, Screen } from '../app';
import { button, chip, progressBar, spinner } from '../components/controls';
import { confirmDialog } from '../components/dialog';
import { toast } from '../components/toast';
import { Disposer, h, replaceChildren } from '../dom';
import { formatClock, formatCount, formatDuration, formatEta, qualityLabel } from '../format';
import { icon } from '../icons';
import type { ProcessingStep, Route } from '../router';
import { STAGES, estimateRemainingMs, stageIdFor, stageWeights, type StageId, type StageState } from '../stages';
import { decimate } from './capture';
import { clearDenseResume as clearResumeMarkers, denseResumable } from '../model-info';

interface StageRow {
  el: HTMLElement;
  bar: ReturnType<typeof progressBar>;
  pct: HTMLElement;
  statusIcon: HTMLElement;
  detail: HTMLElement;
}

const DEFAULT_STAGE: Record<ProcessingStep, StageId> = { refine: 'matching', densify: 'depth_maps', mesh: 'meshing' };

export function createProcessingScreen(app: App, route: Extract<Route, { screen: 'processing' }>): Screen {
  const d = new Disposer();
  const steps: ProcessingStep[] = [...route.steps];
  // Opened from Home, `app.results` still describes the PREVIOUS scan until the engine loaded
  // this one (run → ensureSessionLoaded): name, quality and "already built" stages are then
  // filled in after the load instead of showing the other scan's.
  let resultsStale = !!route.sessionId && app.loadedSessionId !== route.sessionId;
  /** This scan has a sparse model to open (not the previous scan's, while it is not loaded). */
  const hasModel = (): boolean => !resultsStale && !!app.results.sparse;
  let quality = app.results.quality;
  let weights = stageWeights(quality);
  let closed = false;
  let running = false;
  let cancelling = false;
  const startedAt = performance.now();

  const states = new Map<StageId, StageState>();
  /** Stages of steps that are not part of this run: already built earlier ('done', no time) or not requested ('skipped'). */
  function initStates(): void {
    const r = app.results;
    const built: Record<ProcessingStep, boolean> = resultsStale ? { refine: false, densify: false, mesh: false } : { refine: !!r.sparse, densify: !!r.dense, mesh: !!r.mesh };
    for (const s of STAGES) {
      const run = steps.includes(s.step);
      states.set(s.id, { id: s.id, status: run ? 'pending' : built[s.step] ? 'done' : 'skipped', progress: run ? 0 : 1, startedAt: null, endedAt: null });
    }
  }
  initStates();

  /** The dense cloud was (re)built, or refine invalidated the stored depth maps: nothing to resume. */
  function clearDenseResume(): void {
    clearResumeMarkers(app.results.stats);
  }

  // ---- DOM ---------------------------------------------------------------------------------------
  const previewHost = h('div', { class: 'proc-preview-canvas' });
  const previewEmpty = h('div', { class: 'proc-preview-empty' }, spinner('md'), h('span', { text: 'Preparing preview…' }));
  const previewCaption = h('div', { class: 'proc-preview-caption' });
  const preview = h('div', { class: 'proc-preview', 'data-testid': 'processing-preview' }, previewHost, previewEmpty, previewCaption);
  const elapsedEl = h('span', { class: 'proc-elapsed', text: '0:00' });
  const etaEl = h('span', { class: 'proc-eta', text: 'Estimating time…', 'data-testid': 'eta' });
  const resuming = steps[0] === 'densify' && denseResumable(app.results);
  const headlineText = steps.includes('refine') ? 'Reconstructing' : steps.includes('mesh') ? 'Building mesh' : resuming ? 'Resuming dense cloud' : 'Building dense cloud';
  const headline = h('h1', { class: 'proc-title', text: headlineText });
  const rows = new Map<StageId, StageRow>();
  const list = h('ol', { class: 'stage-list' });
  for (const s of STAGES) {
    const bar = progressBar(s.label);
    const pct = h('span', { class: 'stage-pct' });
    const statusIcon = h('span', { class: 'stage-icon' });
    const detail = h('span', { class: 'stage-detail', text: s.detail });
    const el = h(
      'li',
      { class: 'stage', 'data-stage': s.id, 'data-testid': `stage-${s.id}` },
      statusIcon,
      h('div', { class: 'stage-main' }, h('div', { class: 'stage-head' }, h('span', { class: 'stage-label', text: s.label }), pct), detail, bar.el),
    );
    rows.set(s.id, { el, bar, pct, statusIcon, detail });
    list.appendChild(el);
  }
  const titleEl = h('span', { class: 'topbar-title', text: resultsStale ? 'Opening scan…' : app.results.name });
  const subEl = h('span', { class: 'topbar-sub', text: resultsStale ? '' : `${qualityLabel(quality)} quality` });
  const cancelBtn = button({ label: 'Cancel', variant: 'ghost', size: 'sm', testId: 'processing-cancel', onClick: () => void cancel() });
  const errorHost = h('div', { class: 'proc-error-host' });

  const el = h(
    'div',
    { class: 'screen-scroll processing' },
    h(
      'header',
      { class: 'topbar' },
      h('span', { class: 'topbar-spacer' }),
      h('div', { class: 'topbar-center' }, titleEl, subEl),
      cancelBtn,
    ),
    h(
      'div',
      { class: 'content' },
      preview,
      h('div', { class: 'proc-summary' }, headline, h('div', { class: 'proc-times' }, icon('activity', { size: 16 }), elapsedEl, h('span', { class: 'dot-sep', text: '·' }), etaEl)),
      errorHost,
      list,
      h('p', { class: 'proc-note' }, icon('info', { size: 14 }), h('span', { text: 'Keep the app open — iOS pauses processing in the background.' })),
    ),
  );

  function renderStage(id: StageId): void {
    const st = states.get(id)!;
    const r = rows.get(id)!;
    r.el.dataset.status = st.status;
    switch (st.status) {
      case 'skipped':
        r.statusIcon.replaceChildren(icon('dot', { size: 10 }));
        r.pct.replaceChildren(chip('Not requested', 'neutral'));
        r.bar.set(0);
        break;
      case 'pending':
        r.statusIcon.replaceChildren(h('span', { class: 'stage-ring' }));
        r.pct.textContent = '';
        r.bar.set(0);
        break;
      case 'active':
        r.statusIcon.replaceChildren(spinner('sm'));
        r.pct.textContent = st.progress >= 0 ? `${Math.round(st.progress * 100)}%` : '';
        r.bar.set(st.progress);
        break;
      case 'done':
        r.statusIcon.replaceChildren(icon('check', { size: 16 }));
        r.pct.textContent = st.startedAt !== null && st.endedAt !== null ? formatDuration(st.endedAt - st.startedAt) : 'Done';
        r.bar.set(1);
        break;
      case 'error':
        r.statusIcon.replaceChildren(icon('warning', { size: 16 }));
        r.pct.textContent = 'Failed';
        break;
    }
    if (st.message && st.status === 'active') r.detail.textContent = st.message;
  }

  function setActive(id: StageId): void {
    const now = performance.now();
    const order = STAGES.map((s) => s.id);
    const idx = order.indexOf(id);
    // Stages before the active one (in the same run) are done.
    for (let i = 0; i < idx; i++) {
      const s = states.get(order[i])!;
      if (s.status === 'active' || (s.status === 'pending' && steps.includes(STAGES[i].step))) {
        s.status = 'done';
        s.startedAt ??= now;
        s.endedAt = now;
        renderStage(s.id);
      }
    }
    const st = states.get(id)!;
    if (st.status !== 'active') {
      st.status = 'active';
      st.startedAt = now;
      st.progress = 0;
    }
  }

  function onProgress(step: ProcessingStep, e: ProgressEvent): void {
    if (closed) return;
    const id = stageIdFor(e.stage) ?? DEFAULT_STAGE[step];
    const def = STAGES.find((s) => s.id === id)!;
    if (def.step !== step) return; // a stage of another step: ignore
    setActive(id);
    const st = states.get(id)!;
    st.progress = e.progress;
    if (e.message) st.message = e.message;
    renderStage(id);
  }

  function completeStep(step: ProcessingStep): void {
    const now = performance.now();
    for (const s of STAGES) {
      if (s.step !== step) continue;
      const st = states.get(s.id)!;
      if (st.status === 'pending' || st.status === 'active') {
        st.status = 'done';
        st.startedAt ??= now;
        st.endedAt = now;
        st.progress = 1;
        renderStage(s.id);
      }
    }
  }

  function failStep(step: ProcessingStep): void {
    for (const s of STAGES) {
      const st = states.get(s.id)!;
      if (s.step === step && st.status === 'active') {
        st.status = 'error';
        renderStage(s.id);
      }
    }
  }

  function tick(): void {
    elapsedEl.textContent = formatClock(performance.now() - startedAt);
    if (!running) return;
    const eta = estimateRemainingMs([...states.values()].filter((s) => s.status !== 'skipped'), weights, performance.now());
    etaEl.textContent = formatEta(eta);
  }

  // ---- preview -----------------------------------------------------------------------------------
  let viewer: Viewer | null = null;
  let ViewerCtor: typeof Viewer | null = null;
  function ensureViewer(): Viewer | null {
    if (viewer || !app.env.webgl || !ViewerCtor) return viewer;
    try {
      viewer = new ViewerCtor(previewHost, { background: 'transparent', autoFit: 'follow' });
      viewer.setDisplay({ grid: false, edl: true, cameras: true });
    } catch (e) {
      console.warn('[processing] preview unavailable', e);
    }
    return viewer;
  }

  function showPreview(): void {
    if (resultsStale) return; // the previous scan's model: run() shows this one's after the load
    const r = app.results;
    if (r.dense) {
      const v = ensureViewer();
      if (!v) return;
      v.setWorldUp(r.sparse?.metric.gravityAligned ? 'z' : 'auto');
      v.setPointCloud(r.dense);
      if (r.sparse) v.setCameras(r.sparse.cameras.map((c) => ({ pose: c.pose, intrinsics: c.intrinsics })), false);
      previewCaption.textContent = `Dense cloud · ${formatCount(r.dense.count)} points`;
    } else if (r.sparse) {
      const v = ensureViewer();
      if (!v) return;
      v.setWorldUp(r.sparse.metric.gravityAligned ? 'z' : 'auto');
      const pc: PointCloud = r.sparse.points;
      const dec = decimate(pc.positions, pc.colors, 200_000);
      v.setPointCloud({ count: dec.positions.length / 3, positions: dec.positions, colors: dec.colors }, { pointSizePx: 2.5, splats: false });
      v.setCameras(r.sparse.cameras.map((c) => ({ pose: c.pose, intrinsics: c.intrinsics })), false);
      previewCaption.textContent = `Sparse model · ${formatCount(pc.count)} points · ${r.sparse.cameras.length} photos`;
    } else if (r.liveSparse && r.liveSparse.positions.length) {
      const v = ensureViewer();
      if (!v) return;
      v.setWorldUp('auto');
      v.setSparsePoints(r.liveSparse.positions, r.liveSparse.colors);
      if (r.liveCameras) v.setCameras(r.liveCameras, false);
      previewCaption.textContent = 'Live preview from the capture';
    } else return;
    previewEmpty.hidden = true;
  }

  // ---- run ---------------------------------------------------------------------------------------
  async function run(): Promise<void> {
    running = true;
    errorHost.replaceChildren();
    cancelBtn.disabled = false;
    if (!(await app.ensureSessionLoaded(route.sessionId))) {
      running = false;
      if (closed) return;
      // Stopped during the load: leave now (the failure was toasted), not at the 8 s fallback.
      if (cancelling) finishProcessing(true);
      else showError('Couldn’t open the scan', 'The stored data could not be loaded.', false);
      return;
    }
    if (closed) return;
    if (resultsStale) {
      resultsStale = false;
      quality = app.results.quality;
      weights = stageWeights(quality);
      titleEl.textContent = app.results.name;
      subEl.textContent = `${qualityLabel(quality)} quality`;
      initStates();
      for (const st of STAGES) renderStage(st.id);
    }
    showPreview();
    let step: ProcessingStep | null = null;
    try {
      for (const s of steps) {
        // A Stop that came while the previous request was ending (loadSession ignores it, a step
        // past its last cancel check completes normally): the next step must not start.
        if (cancelling) {
          running = false;
          finishProcessing(true);
          return;
        }
        step = s;
        if (s === 'refine') {
          setActive('matching');
          renderStage('matching');
          const sparse = await app.engine.refine((e) => onProgress('refine', e));
          if (closed) return;
          app.results.sparse = sparse;
          app.results.stats = { ...app.results.stats, ...sparse.stats };
          clearDenseResume(); // refine deletes the depth maps of the old poses
          completeStep('refine');
          showPreview();
          void app.updateSession(route.sessionId, { status: 'sparse', stats: numericStats(sparse.stats) });
        } else if (s === 'densify') {
          if (!app.results.sparse) throw new Error('No sparse model to densify');
          setActive('depth_maps');
          renderStage('depth_maps');
          const dense = await app.engine.densify({ quality }, (e) => onProgress('densify', e));
          if (closed) return;
          app.results.dense = dense;
          app.results.mesh = null; // the engine dropped the mesh of the previous cloud
          app.results.stats = { ...app.results.stats, denseQuality: quality };
          clearDenseResume();
          completeStep('densify');
          showPreview();
          void app.updateSession(route.sessionId, { status: 'dense', stats: { densePoints: dense.count, denseQuality: quality } });
        } else {
          setActive('meshing');
          renderStage('meshing');
          const mesh = await app.engine.mesh({ quality }, (e) => onProgress('mesh', e));
          if (closed) return;
          app.results.mesh = mesh;
          app.results.stats = { ...app.results.stats, meshQuality: quality };
          completeStep('mesh');
          void app.updateSession(route.sessionId, { status: 'meshed', stats: { triangles: mesh.triangleCount, meshQuality: quality } });
        }
      }
    } catch (e) {
      running = false;
      // An interrupted densify of a stored scan continues from its saved depth maps next time
      // (the viewer offers "Resume dense reconstruction").
      // The run's preset is kept with the marker: a resume must use it (another one starts over).
      if (step === 'densify' && route.sessionId && !app.results.dense) app.results.stats = { ...app.results.stats, denseInterrupted: 1, denseInterruptedQuality: quality };
      if (closed) return;
      if (step) failStep(step);
      if (isCancelledError(e) || cancelling) {
        finishProcessing(true);
        return;
      }
      const name = (e as { name?: string })?.name;
      const msg = (e as Error)?.message ?? String(e);
      if (app.engine.status.state === 'dead') {
        // Retrying cannot work: EngineClient gave up after repeated worker crashes.
        showError('Processing engine stopped', `${msg}. Your capture is saved — reload the app to process it.`, false);
        return;
      }
      showError(
        name === ENGINE_ERRORS.crashed ? 'The engine restarted' : 'Processing failed',
        name === ENGINE_ERRORS.crashed ? `${msg}. Your capture is saved — try again.` : msg,
        true,
      );
      return;
    }
    running = false;
    if (closed) return;
    etaEl.textContent = 'Done';
    finishProcessing(false);
  }

  function showError(title: string, text: string, canRetry: boolean): void {
    cancelBtn.disabled = true;
    headline.textContent = 'Processing stopped';
    const actions: HTMLElement[] = [];
    if (canRetry) actions.push(button({ label: 'Try again', icon: 'refresh', variant: 'primary', size: 'md', onClick: () => void retry(), testId: 'processing-retry' }));
    actions.push(
      button({
        label: hasModel() ? 'Open model' : 'Back to home',
        variant: 'secondary',
        size: 'md',
        onClick: () => finishProcessing(true),
      }),
    );
    replaceChildren(
      errorHost,
      h(
        'div',
        { class: 'banner banner-bad', role: 'alert', 'data-testid': 'processing-error' },
        h('span', { class: 'banner-icon' }, icon('warning', { size: 20 })),
        h('div', { class: 'banner-text' }, h('div', { class: 'banner-title', text: title }), h('div', { class: 'banner-body', text: text }), h('div', { class: 'banner-actions' }, ...actions)),
      ),
    );
    etaEl.textContent = 'Stopped';
  }

  async function retry(): Promise<void> {
    headline.textContent = headlineText;
    for (const s of STAGES) {
      const st = states.get(s.id)!;
      if (st.status === 'error' || st.status === 'active') {
        st.status = 'pending';
        st.progress = 0;
        st.startedAt = null;
        st.endedAt = null;
        renderStage(s.id);
      }
    }
    // Do not redo finished steps.
    await runRemaining();
  }

  async function runRemaining(): Promise<void> {
    const remaining = steps.filter((s) => STAGES.some((d) => d.step === s && states.get(d.id)!.status !== 'done'));
    if (!remaining.length) {
      finishProcessing(false);
      return;
    }
    steps.splice(0, steps.length, ...remaining);
    await run();
  }

  function finishProcessing(cancelled: boolean): void {
    if (closed) return;
    closed = true;
    void app.wakeLock.release();
    if (!cancelled) app.navigate({ type: 'PROCESSING_DONE' });
    else app.navigate({ type: 'PROCESSING_CANCELLED', hasModel: hasModel() });
  }

  async function cancel(): Promise<void> {
    if (!running) {
      finishProcessing(true);
      return;
    }
    const ok = await confirmDialog({
      title: 'Stop processing?',
      message: steps.includes('refine') ? 'Your capture stays saved; you can process it later from the home screen.' : 'The model built so far is kept.',
      confirmLabel: 'Stop',
      cancelLabel: 'Continue',
      destructive: true,
    });
    if (!ok || closed) return;
    cancelling = true;
    cancelBtn.disabled = true;
    etaEl.textContent = 'Stopping…';
    app.engine.cancel();
    // If the engine does not acknowledge within 8 s, leave anyway.
    d.timeout(() => {
      if (!closed) finishProcessing(true);
    }, 8000);
  }

  return {
    el,
    enter() {
      // Keep the screen on (iOS suspends a hidden page). Requested here too: opening a stored
      // capture from Home navigates inside that tap.
      void app.wakeLock.acquire();
      for (const s of STAGES) renderStage(s.id);
      void loadViewerModule().then(
        (m) => {
          ViewerCtor = m.Viewer;
          if (!closed) showPreview();
        },
        (e) => console.warn('[processing] preview unavailable', e),
      );
      d.interval(tick, 1000);
      let wasHidden = false;
      d.listen(document, 'visibilitychange', () => {
        if (document.hidden) wasHidden = true;
        else if (wasHidden && running) {
          wasHidden = false;
          toast('Processing continues — it pauses while the app is in the background.', { tone: 'info' });
        }
      });
      void run();
    },
    leave() {
      closed = true;
      viewer?.dispose();
      viewer = null;
      d.dispose();
    },
  };
}

function numericStats(s: Record<string, number | string>): Record<string, number | string> {
  const out: Record<string, number | string> = {};
  for (const [k, v] of Object.entries(s)) if (typeof v === 'number' || (typeof v === 'string' && v.length < 64)) out[k] = v;
  return out;
}
