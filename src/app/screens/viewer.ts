/**
 * Viewer: full-screen 3D model (photoreal Gaussian splats / points / mesh — one layer at a time,
 * switched by the layer pill above the toolbar —, cameras, EDL), measure tool (metres only for
 * metric models; on the points / mesh also while the splats are shown), Build sheet (dense cloud
 * / mesh with a quality picker, and the photoreal model: `engine.splat()` runs HERE with a
 * progress card and Cancel, the model is shown when it finishes), Export sheet (PLY points, PLY
 * mesh, GLB, Gaussian splats PLY, COLMAP + images ZIP: the first tap builds the file through the
 * engine and shows its size, a second tap "Share / Save" calls saveFile directly in the tap
 * handler — iOS allows navigator.share only within ~5 s of a user activation), Info sheet
 * (scale source and uncertainty, gravity / north alignment with the yaw source — the compass
 * gives MAGNETIC north —, counts, reprojection error, engine memory, timings). A densify that
 * was interrupted (stats.denseResumablePass from loadSession, or cancelled / failed here) offers
 * "Resume dense reconstruction" (HUD + Build sheet): the engine continues from its stored maps.
 */
import { saveFile } from '../../io/download';
import type { SplatDisplayState, Viewer } from '../../viewer';
import { loadViewerModule } from '../viewer-loader';
import { formatDistance } from '../../viewer/helpers';
import type { ProgressEvent } from '../../core/types';
import { isCancelledError, type ExportFormat, type QualityPreset } from '../../workers/protocol';
import type { App, Screen } from '../app';
import { button, chip, group, iconButton, kvRow, progressBar, section, segmented, setButtonLabel, spinner, switchRow } from '../components/controls';
import { confirmDialog } from '../components/dialog';
import { openSheet, type SheetHandle } from '../components/sheet';
import { toast } from '../components/toast';
import { Disposer, h, replaceChildren } from '../dom';
import { exportFilename, formatBytes, formatCount, formatDuration, modeLabel, qualityLabel } from '../format';
import { icon, type IconName } from '../icons';
import type { ProcessingStep, Route } from '../router';
import {
  builtQuality,
  clearDenseResume,
  clearSplatResume,
  denseResumable,
  denseResumeQuality,
  memoryStatRows,
  northInfo,
  splatProgressView,
  splatResumable,
  splatResumeQuality,
} from '../model-info';

/** What the 3D view shows: one model layer at a time. */
type LayerId = 'splats' | 'points' | 'mesh';

interface DisplayState {
  layer: LayerId;
  cameras: boolean;
  edl: boolean;
  grid: boolean;
  /** Light the mesh (default off: its colours are photo colours, which already contain the shading). */
  meshLighting: boolean;
}

const LAYER_LABELS: Record<LayerId, string> = { splats: 'Splats', points: 'Points', mesh: 'Mesh' };

interface ExportState {
  status: 'idle' | 'building' | 'ready' | 'error';
  blob?: Blob;
  error?: string;
}

const EXPORTS: { format: ExportFormat; title: string; icon: IconName; ext: string; suffix: string }[] = [
  { format: 'ply-points', title: 'Point cloud (PLY)', icon: 'points', ext: 'ply', suffix: 'points' },
  { format: 'ply-mesh', title: 'Mesh (PLY)', icon: 'mesh', ext: 'ply', suffix: 'mesh' },
  { format: 'glb', title: '3D model (GLB)', icon: 'cube', ext: 'glb', suffix: '' },
  { format: 'ply-splat', title: 'Gaussian splats (.ply)', icon: 'splats', ext: 'ply', suffix: 'splats' },
  { format: 'colmap-zip', title: 'COLMAP project (ZIP)', icon: 'zip', ext: 'zip', suffix: 'colmap' },
];

export function createViewerScreen(app: App, route: Extract<Route, { screen: 'viewer' }>): Screen {
  const d = new Disposer();
  let viewer: Viewer | null = null;
  let closed = false;
  let measuring = false;
  const display: DisplayState = { layer: 'points', cameras: true, edl: true, grid: true, meshLighting: false };
  const exports = new Map<ExportFormat, ExportState>();
  /** Bumped when a photoreal build replaces the model: a splat PLY prepared before is stale. */
  let splatVersion = 0;
  /** The viewer's splat layer ('none' until a model is set; 'unavailable': no WebGPU / failed). */
  let splatState: SplatDisplayState = 'none';
  /** engine.splatModel() is being fetched. */
  let splatFetching = false;
  /** Bumped by every splat model load (a slower earlier one must not install its model). */
  let splatLoadToken = 0;
  /** The user picked a layer: a model finishing to load does not switch the view on its own. */
  let userPickedLayer = false;
  /** A photoreal build (engine.splat) runs from this screen. */
  let building = false;
  let cancelling = false;
  /** Splats cannot be built here (no WebGPU in this browser; the engine would reject). */
  const webgpuMissing = (): boolean => app.env.webgpu === 'unsupported' || app.env.webgpu === 'no-adapter';

  const canvasHost = h('div', { class: 'viewer-canvas', 'data-testid': 'viewer-canvas' });
  const loading = h('div', { class: 'viewer-loading' }, spinner('lg'), h('span', { text: 'Loading model…' }));
  // Opened from Home, app.results may still hold the previously loaded scan until load() ran.
  const titleEl = h('span', { class: 'viewer-title', text: route.sessionId && app.loadedSessionId !== route.sessionId ? '' : app.results.name });
  const subEl = h('span', { class: 'viewer-sub' });
  const top = h(
    'header',
    { class: 'viewer-top' },
    iconButton('back', 'Back to scans', () => void goBack(), { variant: 'glass', testId: 'viewer-back' }),
    h('div', { class: 'viewer-heading' }, titleEl, subEl),
    iconButton('info', 'Model info', () => openInfo(), { variant: 'glass', testId: 'viewer-info' }),
  );

  const measureValue = h('div', { class: 'measure-value', 'data-testid': 'measure-value' });
  const measureHint = h('div', { class: 'measure-hint', text: 'Tap two points on the model' });
  const measureHud = h(
    'div',
    { class: 'measure-hud', hidden: true, role: 'status', 'aria-live': 'polite' },
    h('div', { class: 'measure-text' }, measureValue, measureHint),
    button({ label: 'Done', variant: 'glass', size: 'sm', onClick: () => setMeasure(false) }),
  );

  const tool = (name: IconName, label: string, onClick: () => void, testId: string): HTMLButtonElement => {
    const b = h('button', { type: 'button', class: 'tool', 'aria-label': label, 'data-testid': testId }, icon(name, { size: 22 }), h('span', { class: 'tool-label', text: label }));
    b.addEventListener('click', onClick);
    return b;
  };
  const viewTool = tool('layers', 'View', () => openDisplay(), 'tool-view');
  const measureTool = tool('ruler', 'Measure', () => setMeasure(!measuring), 'tool-measure');
  measureTool.setAttribute('aria-pressed', 'false');
  const buildTool = tool('sparkles', 'Build', () => openBuild(), 'tool-build');
  const exportTool = tool('share', 'Export', () => openExport(), 'tool-export');
  const toolbar = h('nav', { class: 'viewer-toolbar', 'aria-label': 'Model tools' }, viewTool, measureTool, buildTool, exportTool);

  const resumeSub = h('span', { class: 'viewer-resume-sub', text: 'Resume from the saved depth maps.' });
  const resumeHud = h(
    'div',
    { class: 'viewer-resume', hidden: true, role: 'status', 'data-testid': 'dense-resume' },
    h('span', { class: 'viewer-resume-icon' }, icon('points', { size: 20 })),
    h('div', { class: 'viewer-resume-text' }, h('span', { class: 'viewer-resume-title', text: 'Dense cloud interrupted' }), resumeSub),
    button({ label: 'Resume', variant: 'primary', size: 'sm', testId: 'dense-resume-button', onClick: () => resumeDense() }),
  );

  // Interrupted photoreal build: the same card for the splats (dense takes precedence).
  const splatResumeSub = h('span', { class: 'viewer-resume-sub' });
  const splatResumeHud = h(
    'div',
    { class: 'viewer-resume', hidden: true, role: 'status', 'data-testid': 'splat-resume' },
    h('span', { class: 'viewer-resume-icon' }, icon('splats', { size: 20 })),
    h('div', { class: 'viewer-resume-text' }, h('span', { class: 'viewer-resume-title', text: 'Photoreal model interrupted' }), splatResumeSub),
    button({ label: 'Resume', variant: 'primary', size: 'sm', testId: 'splat-resume-button', onClick: () => void buildSplats(splatResumeQuality(app.results) ?? builtQuality(app.results, 'splat')) }),
  );

  // Photoreal build progress (engine.splat runs while the model stays explorable).
  const jobBar = progressBar('Photoreal model');
  const jobSub = h('span', { class: 'viewer-job-sub', 'data-testid': 'splat-progress-text' });
  const jobCancel = button({ label: 'Cancel', variant: 'glass', size: 'sm', testId: 'splat-cancel', onClick: () => void cancelBuild() });
  const jobHud = h(
    'div',
    { class: 'viewer-job', hidden: true, role: 'status', 'aria-live': 'polite', 'data-testid': 'splat-progress' },
    h('span', { class: 'viewer-job-icon' }, spinner('sm', 'Building')),
    h('div', { class: 'viewer-job-text' }, h('span', { class: 'viewer-job-title', text: 'Building photoreal model' }), jobSub, jobBar.el),
    jobCancel,
  );

  // Layer pill above the toolbar (Splats / Points / Mesh), shown when there is a choice.
  const layerHost = h('div', { class: 'layer-switch', hidden: true, 'data-testid': 'layer-switch' });

  const el = h('div', { class: 'viewer-screen' }, canvasHost, loading, top, resumeHud, splatResumeHud, jobHud, measureHud, layerHost, toolbar);

  /**
   * Resume (re-run) the interrupted densify with the quality of the INTERRUPTED run (the engine skips
   * the stored maps only for the same settings), not the scan's current one.
   */
  function resumeDense(): void {
    const rq = denseResumeQuality(app.results);
    if (rq) app.results.quality = rq;
    app.navigate({ type: 'BUILD', steps: ['densify'] });
  }

  // ---- content -----------------------------------------------------------------------------------
  /** Splats can be shown: a trained model exists and the viewer did not give up on its layer. */
  function splatsAvailable(): boolean {
    return !!app.results.splat && splatState !== 'unavailable' && !!viewer;
  }

  function availableLayers(): LayerId[] {
    const r = app.results;
    const out: LayerId[] = [];
    if (splatsAvailable()) out.push('splats');
    if (r.dense || r.sparse) out.push('points');
    if (r.mesh) out.push('mesh');
    return out;
  }

  /** The layer to show when the current one is gone (splats → mesh → points). */
  function fallbackLayer(): LayerId {
    const av = availableLayers();
    if (av.includes(display.layer)) return display.layer;
    return av.includes('mesh') ? 'mesh' : 'points';
  }

  function applyContent(first: boolean): void {
    const r = app.results;
    titleEl.textContent = r.name;
    renderBadges();
    if (!viewer) {
      renderSub();
      return;
    }
    const metric = r.sparse?.metric;
    viewer.setWorldUp(metric?.gravityAligned ? 'z' : 'auto');
    viewer.setUnits(!!metric?.metric);
    if (r.dense) viewer.setPointCloud(r.dense);
    else if (r.sparse) viewer.setPointCloud(r.sparse.points, { pointSizePx: 3, splats: false });
    else viewer.setPointCloud(null);
    viewer.setMesh(r.mesh);
    viewer.setCameras(r.sparse ? r.sparse.cameras.map((c) => ({ pose: c.pose, intrinsics: c.intrinsics })) : [], false);
    if (first) {
      // the photoreal model when there is one (points meanwhile, while it loads), else the mesh
      display.layer = r.splat ? 'splats' : r.mesh ? 'mesh' : 'points';
      display.grid = !!metric?.gravityAligned;
      // splats have no depth for the frustums to hide behind: they would cover the view
      display.cameras = !r.splat;
    }
    if (display.layer !== 'splats' || !r.splat) display.layer = fallbackLayer();
    applyDisplay();
    if (first) frameView(false);
    if (r.splat && splatState === 'none' && !splatFetching) void loadSplats(false);
  }

  /** Header sub line: the shown layer's size and the photo count. */
  function renderSub(): void {
    const r = app.results;
    const parts: string[] = [];
    const layer = display.layer === 'splats' && splatState !== 'ready' ? 'points' : display.layer;
    if (layer === 'splats' && r.splat) parts.push(`${formatCount(r.splat.count)} Gaussians`);
    else if (layer === 'mesh' && r.mesh) parts.push(`${formatCount(r.mesh.triangleCount)} triangles`);
    else if (r.dense) parts.push(`${formatCount(r.dense.count)} points`);
    else if (r.sparse) parts.push(`${formatCount(r.sparse.points.count)} points`);
    if (r.sparse) parts.push(`${r.sparse.cameras.length} photos`);
    subEl.textContent = parts.join(' · ');
  }

  /** Build tool badge (something left to build) and the resume cards. */
  function renderBadges(): void {
    const r = app.results;
    buildTool.classList.toggle('has-badge', !r.dense || !r.mesh || (!r.splat && !!r.sparse && !webgpuMissing()));
    const dense = denseResumable(r);
    resumeHud.hidden = !dense || building;
    const rq = denseResumeQuality(r);
    resumeSub.textContent = rq ? `Resume from the saved depth maps (${qualityLabel(rq)} quality).` : 'Resume from the saved depth maps.';
    splatResumeHud.hidden = dense || building || !splatResumable(r) || webgpuMissing();
    const sq = splatResumeQuality(r);
    const it = r.stats.splatResumableIteration;
    splatResumeSub.textContent = `Continues from the saved training${typeof it === 'number' && it > 0 ? ` (iteration ${it.toLocaleString('en-US')})` : ''}${sq ? ` · ${qualityLabel(sq)}` : ''}.`;
    jobHud.hidden = !building;
    el.classList.toggle('building', building);
  }

  function applyDisplay(): void {
    const l = display.layer;
    viewer?.setDisplay({
      // splats replace the points / mesh once ready; the points show while they load
      splats: l === 'splats',
      points: l !== 'mesh',
      mesh: l === 'mesh',
      cameras: display.cameras,
      edl: display.edl,
      grid: display.grid,
      unlit: !display.meshLighting,
    });
    renderLayerSwitch();
    renderSub();
  }

  /**
   * Default view: with splats, from the middle photo (they are photoreal near the capture and show
   * floaters far outside it); otherwise the whole content.
   */
  function frameView(animate: boolean): void {
    if (!viewer) return;
    if (display.layer === 'splats' && viewer.viewFromCamera(undefined, { animate })) return;
    viewer.fitToContent(animate);
  }

  function setDisplay(patch: Partial<DisplayState>): void {
    Object.assign(display, patch);
    applyDisplay();
  }

  function setLayer(l: LayerId): void {
    userPickedLayer = true;
    setDisplay({ layer: l });
  }

  let layerSegments: { el: HTMLElement; set: (v: LayerId) => void } | null = null;
  let layerKey = '';
  function renderLayerSwitch(): void {
    const av = availableLayers();
    const loadingSplats = display.layer === 'splats' && (splatFetching || splatState === 'loading' || splatState === 'lost');
    layerHost.hidden = av.length < 2;
    layerHost.classList.toggle('loading', loadingSplats);
    layerHost.dataset.layer = display.layer;
    layerHost.dataset.splats = splatFetching ? 'fetching' : splatState;
    const key = av.join(',');
    if (key !== layerKey || !layerSegments) {
      layerKey = key;
      layerSegments = segmented<LayerId>(
        'Model layer',
        av.map((v) => ({ value: v, label: LAYER_LABELS[v] })),
        av.includes(display.layer) ? display.layer : av[0],
        (v) => setLayer(v),
      );
      replaceChildren(layerHost, layerSegments.el, h('span', { class: 'layer-status', 'aria-hidden': 'true' }, spinner('sm', 'Loading photoreal model')));
    } else layerSegments.set(display.layer);
  }

  /**
   * Fetches the stored splat model (transferred from the engine) into the viewer's WebGPU layer.
   * `show`: switch to it when ready (after a build); otherwise only when the user did not pick
   * another layer meanwhile.
   */
  async function loadSplats(show: boolean): Promise<void> {
    if (!viewer || !app.results.splat) return;
    const token = ++splatLoadToken;
    splatFetching = true;
    renderLayerSwitch();
    try {
      const payload = await app.engine.splatModel();
      if (closed || token !== splatLoadToken || !viewer) return;
      splatFetching = false;
      if (!payload) {
        // the engine has none (e.g. refine removed it)
        app.results.splat = null;
        display.layer = fallbackLayer();
        applyDisplay();
        return;
      }
      const ok = await viewer.setSplats(payload);
      if (closed || token !== splatLoadToken) return;
      if (ok && (show || !userPickedLayer)) display.layer = 'splats';
      if (ok && show) {
        // a model just built: show it from the photos, without the frustums in front of it
        display.cameras = false;
        frameView(true);
      }
      if (!ok && display.layer === 'splats') display.layer = fallbackLayer();
      applyDisplay();
    } catch (e) {
      if (closed || token !== splatLoadToken) return;
      splatFetching = false;
      console.warn('[viewer] splat model unavailable', e);
      toast(`Couldn’t load the photoreal model: ${(e as Error)?.message ?? e}`, { tone: 'warn' });
      if (display.layer === 'splats') display.layer = fallbackLayer();
      applyDisplay();
    }
  }

  let splatUnavailableToast = false;
  function onSplatState(st: SplatDisplayState, detail?: string): void {
    splatState = st;
    if (closed) return;
    el.dataset.splats = st;
    if (st === 'unavailable') {
      if (!splatUnavailableToast) {
        splatUnavailableToast = true;
        toast(
          webgpuMissing() || /software|WebGPU is not available/i.test(detail ?? '')
            ? 'The photoreal model needs WebGPU to be shown — showing the points instead.'
            : `The photoreal model can’t be shown right now (${detail ?? 'GPU error'}).`,
          { tone: 'warn', durationMs: 6000 },
        );
      }
      if (display.layer === 'splats') display.layer = fallbackLayer();
    }
    applyDisplay();
  }

  // ---- measure -----------------------------------------------------------------------------------
  function setMeasure(on: boolean): void {
    if (!viewer) return;
    measuring = on;
    measureTool.classList.toggle('active', on);
    measureTool.setAttribute('aria-pressed', String(on));
    measureHud.hidden = !on;
    el.classList.toggle('measuring', on);
    if (on) {
      measureValue.textContent = '';
      measureHint.textContent = 'Tap two points on the model';
      viewer.startMeasure((dist) => {
        const metric = app.results.sparse?.metric;
        const isMetric = !!metric?.metric;
        let txt = formatDistance(dist, isMetric);
        const unc = metric?.relativeUncertainty;
        if (isMetric && unc && unc > 0) txt += ` ±${formatDistance(dist * unc, true)}`;
        measureValue.textContent = txt;
        measureHint.textContent = isMetric ? 'Tap again to measure another distance' : 'No real-world scale for this scan — units are arbitrary';
      });
    } else viewer.stopMeasure();
  }

  // ---- sheets ------------------------------------------------------------------------------------
  function openDisplay(): void {
    const r = app.results;
    const av = availableLayers();
    const layers = segmented<LayerId>(
      'Show',
      av.map((v) => ({ value: v, label: LAYER_LABELS[v] })),
      av.includes(display.layer) ? display.layer : av[0],
      (v) => setLayer(v),
    );
    layers.el.dataset.testid = 'display-layer';
    const note = !r.splat
      ? 'Splats: build the photoreal model (Build).'
      : splatState === 'unavailable'
        ? 'Splats need WebGPU, which is not available here.'
        : 'Splats are the photoreal model; measuring uses the points and mesh.';
    const rows = [
      switchRow({ label: 'Cameras', icon: 'frustum', checked: display.cameras, testId: 'display-cameras', onChange: (v) => setDisplay({ cameras: v }) }),
      switchRow({ label: 'Eye-dome lighting', icon: 'sun', description: 'Shades point clouds for depth', checked: display.edl, testId: 'display-edl', onChange: (v) => setDisplay({ edl: v }) }),
      switchRow({
        label: 'Mesh lighting',
        icon: 'mesh',
        description: 'Off: the photo colours as captured (they already contain the light)',
        checked: display.meshLighting,
        disabled: !r.mesh,
        testId: 'display-mesh-lighting',
        onChange: (v) => setDisplay({ meshLighting: v }),
      }),
      switchRow({ label: 'Ground grid', icon: 'target', description: 'Not with splats', checked: display.grid, disabled: !r.sparse?.metric.gravityAligned, onChange: (v) => setDisplay({ grid: v }) }),
    ];
    const reset = button({ label: 'Reset view', icon: 'target', variant: 'secondary', size: 'md', block: true, onClick: () => frameView(true) });
    openSheet({
      title: 'Display',
      content: [section('Show', layers.el, h('p', { class: 'section-note', text: note })), group(...rows.map((r) => r.el)), h('div', { class: 'sheet-footer' }, reset)],
      testId: 'sheet-display',
    });
  }

  function openBuild(): void {
    const r = app.results;
    const resumable = denseResumable(r);
    // the interrupted run's preset (null: unknown — older records)
    const resumeQ = resumable ? denseResumeQuality(r) : null;
    let q: QualityPreset = resumeQ ?? app.results.quality;
    // Resuming needs the interrupted run's settings: another quality starts over; a model built
    // at another quality than the picked one can be rebuilt.
    const syncs: (() => void)[] = [];
    const quality = segmented<QualityPreset>(
      'Quality',
      [
        { value: 'fast', label: 'Fast' },
        { value: 'balanced', label: 'Balanced' },
        { value: 'max', label: 'Max' },
      ],
      q,
      (v) => {
        q = v;
        for (const f of syncs) f();
      },
    );
    const build = (steps: ProcessingStep[]): void => {
      sheet.close();
      app.results.quality = q;
      app.navigate({ type: 'BUILD', steps });
    };
    const item = (o: {
      icon: IconName;
      title: string;
      /** What was built (null: not built yet) and at which quality. */
      done: { sub: string; quality: QualityPreset } | null;
      desc: string;
      /** Sub line when rebuilding at quality `q`. */
      rebuildDesc: (q: QualityPreset) => string;
      steps: ProcessingStep[];
      /** Runs here instead of the Processing screen (the photoreal build). */
      run?: (q: QualityPreset) => void;
      /** Resumable at this quality: "Resume" + this sub line. */
      resume?: (q: QualityPreset) => string | null;
      testId: string;
      disabled?: boolean;
    }): HTMLElement => {
      const sub = h('span', { class: 'build-sub' });
      const action = h('div', { class: 'build-action' });
      const sync = (): void => {
        const built = o.done && o.done.quality === q;
        const resume = o.resume?.(q) ?? null;
        sub.textContent = resume ?? (!o.done ? o.desc : built ? `${o.done.sub} · ${qualityLabel(o.done.quality)}` : o.rebuildDesc(q));
        const onClick = (): void => {
          if (!o.run) return build(o.steps);
          sheet.close();
          o.run(q);
        };
        replaceChildren(
          action,
          built && !resume
            ? chip('Built', 'good', 'check')
            : button({ label: resume ? 'Resume' : o.done ? 'Rebuild' : 'Build', variant: 'primary', size: 'sm', testId: o.testId, disabled: o.disabled, onClick }),
        );
      };
      syncs.push(sync);
      sync();
      return h('div', { class: 'build-item' }, h('span', { class: 'build-icon' }, icon(o.icon, { size: 22 })), h('div', { class: 'build-text' }, h('span', { class: 'build-title', text: o.title }), sub), action);
    };
    const content = [
      section('Quality', quality.el),
      section(
        null,
        h(
          'div',
          { class: 'build-list' },
          resumable
            ? resumeItem()
            : item({
                icon: 'points',
                title: 'Dense point cloud',
                done: r.dense ? { sub: `${formatCount(r.dense.count)} points`, quality: builtQuality(r, 'dense') } : null,
                desc: 'Multi-view stereo on every photo. A few minutes.',
                rebuildDesc: (q) => `Rebuilds at ${qualityLabel(q)} quality${r.mesh ? ' and removes the mesh' : ''}.`,
                steps: ['densify'],
                testId: 'build-dense',
                disabled: !r.sparse || building,
              }),
          item({
            icon: 'mesh',
            title: 'Mesh',
            done: r.mesh ? { sub: `${formatCount(r.mesh.triangleCount)} triangles`, quality: builtQuality(r, 'mesh') } : null,
            desc: r.dense ? 'A coloured surface from the dense cloud.' : 'Builds the dense cloud first, then the surface.',
            rebuildDesc: (q) => `Rebuilds the dense cloud and mesh at ${qualityLabel(q)} quality.`,
            steps: r.dense ? ['mesh'] : ['densify', 'mesh'],
            testId: 'build-mesh',
            disabled: !r.sparse || building,
          }),
          item({
            icon: 'splats',
            title: 'Photoreal model',
            done: r.splat ? { sub: `${formatCount(r.splat.count)} Gaussians`, quality: builtQuality(r, 'splat') } : null,
            desc: webgpuMissing()
              ? 'Needs WebGPU, which this browser does not have.'
              : `Gaussian splats trained on your photos — looks like the real scene.${r.dense ? '' : ' Better after the dense cloud.'} A few minutes.`,
            rebuildDesc: (q) => `Retrains at ${qualityLabel(q)} quality.`,
            steps: [],
            run: (q) => void buildSplats(q),
            resume: (q) => {
              if (!splatResumable(r)) return null;
              const rq = splatResumeQuality(r);
              if (rq && rq !== q) return null;
              return 'Interrupted — continues from the saved training.';
            },
            testId: 'build-splat',
            disabled: !r.sparse || building || webgpuMissing(),
          }),
        ),
      ),
    ];
    const sheet = openSheet({ title: 'Build', subtitle: 'Higher quality takes longer and needs more memory.', content, testId: 'sheet-build' });

    function resumeItem(): HTMLElement {
      const quality0 = resumeQ ?? r.quality;
      const sub = h('span', { class: 'build-sub' });
      const btn = button({
        label: 'Resume',
        variant: 'primary',
        size: 'sm',
        testId: 'build-dense',
        // like the other builds: not while the photoreal model trains (leaving would stop it)
        disabled: building,
        onClick: () => {
          // Another quality starts over (the engine discards the stored maps of the other settings).
          if (q !== quality0) clearDenseResume(app.results.stats);
          build(['densify']);
        },
      });
      const title = h('span', { class: 'build-title' });
      const syncResume = (): void => {
        const same = q === quality0;
        title.textContent = same ? 'Resume dense reconstruction' : 'Dense point cloud';
        sub.textContent = !same
          ? `Starts over at ${qualityLabel(q)} quality (the interrupted run used ${qualityLabel(quality0)}).`
          : resumeQ
            ? 'Interrupted — continues from the saved depth maps.'
            : 'Interrupted — continues from the saved depth maps if they were made at this quality, otherwise starts over.';
        setButtonLabel(btn, same ? 'Resume' : 'Build');
      };
      syncs.push(syncResume);
      syncResume();
      return h('div', { class: 'build-item build-item-resume', 'data-testid': 'build-dense-resume' }, h('span', { class: 'build-icon' }, icon('refresh', { size: 22 })), h('div', { class: 'build-text' }, title, sub), btn);
    }
  }

  function exportAvailability(f: ExportFormat): { ok: boolean; sub: string } {
    const r = app.results;
    switch (f) {
      case 'ply-points':
        if (r.dense) return { ok: true, sub: `Dense cloud · ${formatCount(r.dense.count)} points, normals, colours` };
        if (r.sparse) return { ok: true, sub: `Sparse points · ${formatCount(r.sparse.points.count)}` };
        return { ok: false, sub: 'No points yet' };
      case 'ply-mesh':
        return r.mesh ? { ok: true, sub: `${formatCount(r.mesh.triangleCount)} triangles with vertex colours` } : { ok: false, sub: 'Build the mesh first' };
      case 'glb':
        if (r.mesh) return { ok: true, sub: 'Coloured mesh — opens in most 3D apps and AR' };
        if (r.dense || r.sparse) return { ok: true, sub: 'Coloured point cloud (glTF binary)' };
        return { ok: false, sub: 'Nothing to export yet' };
      case 'colmap-zip':
        return r.sparse ? { ok: true, sub: `Cameras, sparse model and ${r.sparse.cameras.length} photos` } : { ok: false, sub: 'No sparse model yet' };
      case 'ply-splat':
        return r.splat
          ? { ok: true, sub: `${formatCount(r.splat.count)} Gaussians · opens in SuperSplat, Spark and most splat viewers` }
          : { ok: false, sub: 'Build the photoreal model first' };
    }
  }

  /** Re-renders the OPEN export sheet (null when closed): a build that finishes after the sheet
   *  was closed and opened again must update the new sheet, not the detached old list. */
  let renderExports: (() => void) | null = null;
  /** Until this time (or the share settles) further Save taps are ignored: a double tap would
   *  reject (InvalidStateError) and, in a Safari tab, fall back to a download on top of the share
   *  sheet. Time-limited because a share promise that never settles must not disable the button. */
  let sharingUntil = 0;

  function openExport(): void {
    const list = h('div', { class: 'export-list' });
    const render = (): void => {
      replaceChildren(list, ...EXPORTS.map((e) => exportRow(e)));
    };
    const sheet: SheetHandle = openSheet({
      title: 'Export',
      subtitle: 'Files are created on this device.',
      content: list,
      testId: 'sheet-export',
      onClose: () => {
        if (renderExports === render) renderExports = null;
      },
    });
    renderExports = render;
    render();
    void sheet;
  }

  function exportRow(e: (typeof EXPORTS)[number]): HTMLElement {
    const av = exportAvailability(e.format);
    const st = exports.get(e.format) ?? { status: 'idle' };
    const filename = exportFilename(app.results.name, e.ext, e.suffix);
    let action: HTMLElement;
    let sub = av.sub;
    if (!av.ok) action = chip('Unavailable', 'neutral');
    else if (st.status === 'building') action = h('span', { class: 'export-building' }, spinner('sm'), h('span', { text: 'Preparing…' }));
    else if (st.status === 'ready' && st.blob) {
      sub = `${filename} · ${formatBytes(st.blob.size)}`;
      const blob = st.blob;
      action = button({
        label: app.env.share ? 'Share / Save' : 'Save',
        icon: app.env.share ? 'share' : 'download',
        variant: 'primary',
        size: 'sm',
        testId: `export-save-${e.format}`,
        // Fresh user activation: saveFile is called synchronously in this tap.
        onClick: () => {
          if (performance.now() < sharingUntil) return;
          sharingUntil = performance.now() + 2000;
          saveFile(blob, filename)
            .then(
              (res) => {
                if (res === 'shared') toast('Shared', { tone: 'good' });
                else if (res === 'downloaded') toast(`Saved ${filename}`, { tone: 'good', icon: 'download' });
              },
              (err) => {
                const n = (err as { name?: string })?.name;
                if (n === 'NotAllowedError') toast(`Tap “${app.env.share ? 'Share / Save' : 'Save'}” again to save the file.`, { tone: 'warn' });
                else if (n !== 'InvalidStateError') toast(`Could not save: ${(err as Error)?.message ?? err}`, { tone: 'bad' });
              },
            )
            .finally(() => {
              sharingUntil = 0;
            });
        },
      });
    } else
      action = button({
        label: st.status === 'error' ? 'Retry' : 'Prepare',
        variant: 'secondary',
        size: 'sm',
        testId: `export-prepare-${e.format}`,
        onClick: () => void prepare(e.format),
      });
    if (st.status === 'error' && st.error) sub = st.error;
    return h(
      'div',
      { class: `export-item${!av.ok ? ' disabled' : ''}${st.status === 'error' ? ' error' : ''}`, 'data-testid': `export-${e.format}` },
      h('span', { class: 'export-icon' }, icon(e.icon, { size: 22 })),
      h('div', { class: 'export-text' }, h('span', { class: 'export-title', text: e.title }), h('span', { class: 'export-sub', text: sub })),
      action,
    );
  }

  async function prepare(format: ExportFormat): Promise<void> {
    const version = splatVersion;
    exports.set(format, { status: 'building' });
    renderExports?.();
    try {
      const blob = await app.engine.exportModel(format);
      // a photoreal build replaced the model meanwhile: the file may hold the previous one
      if (format === 'ply-splat' && version !== splatVersion) exports.delete(format);
      else exports.set(format, { status: 'ready', blob });
    } catch (e) {
      exports.set(format, { status: 'error', error: (e as Error)?.message ?? String(e) });
    }
    if (!closed) renderExports?.();
  }

  function openInfo(): void {
    const r = app.results;
    const sp = r.sparse;
    const m = sp?.metric;
    const st = { ...(sp?.stats ?? {}), ...r.stats };
    const num = (k: string): number | null => (typeof st[k] === 'number' ? (st[k] as number) : null);
    const yes = (b: boolean | undefined, y: string, n: string): HTMLElement => chip(b ? y : n, b ? 'good' : 'neutral', b ? 'check' : undefined);
    const scaleText = m?.metric ? `Metric · ${m.source === 'imu+gps' ? 'IMU + GPS' : m.source.toUpperCase()}` : 'Not metric (arbitrary units)';
    const rows: HTMLElement[] = [
      kvRow('Scale', h('span', { text: scaleText })),
      kvRow('Scale uncertainty', h('span', { text: m?.metric && m.relativeUncertainty !== null ? `±${(m.relativeUncertainty * 100).toFixed(1)} %` : '–' })),
      kvRow('Up direction', yes(m?.gravityAligned, 'Gravity aligned', 'Not aligned')),
    ];
    const north = northInfo(m?.northAligned, st);
    rows.push(kvRow('North', chip(north.label, north.tone, north.tone === 'good' ? 'check' : north.tone === 'warn' ? 'compass' : undefined)));
    if (north.note) rows.push(h('div', { class: 'row row-note', 'data-testid': 'info-north-note' }, h('span', { class: 'row-sub', text: north.note })));
    if (m?.originLatLonAlt) {
      const [lat, lon, alt] = m.originLatLonAlt;
      rows.push(kvRow('Origin', h('span', { class: 'mono', text: `${lat.toFixed(6)}, ${lon.toFixed(6)} · ${alt.toFixed(0)} m` })));
    }
    const counts: HTMLElement[] = [
      kvRow('Photos', h('span', { text: sp ? `${num('registeredImages') ?? sp.cameras.length} registered of ${num('images') ?? sp.cameras.length}` : '–' })),
      kvRow('Sparse points', h('span', { text: sp ? formatCount(sp.points.count) : '–' })),
      kvRow('Dense points', h('span', { text: r.dense ? formatCount(r.dense.count) : 'Not built' })),
      kvRow('Mesh', h('span', { text: r.mesh ? `${formatCount(r.mesh.triangleCount)} triangles` : 'Not built' })),
      kvRow('Photoreal model', h('span', { text: r.splat ? `${formatCount(r.splat.count)} Gaussians · ${r.splat.iterations.toLocaleString('en-US')} iterations` : 'Not built' })),
      kvRow('Mean reprojection error', h('span', { text: num('meanReprojErrorPx') !== null ? `${num('meanReprojErrorPx')!.toFixed(2)} px` : '–' })),
    ];
    if (num('meanTrackLength') !== null) counts.push(kvRow('Mean track length', h('span', { text: num('meanTrackLength')!.toFixed(1) })));
    if (num('focalPx') !== null) counts.push(kvRow('Focal length', h('span', { text: `${num('focalPx')!.toFixed(1)} px` })));
    if (num('k1') !== null) counts.push(kvRow('Distortion k1 / k2', h('span', { class: 'mono', text: `${num('k1')!.toFixed(4)} / ${(num('k2') ?? 0).toFixed(4)}` })));
    const timings = Object.entries(st)
      .filter(([k, v]) => /Ms$/.test(k) && typeof v === 'number' && k !== 'durationMs')
      .map(([k, v]) => kvRow(timingLabel(k), h('span', { text: formatDuration(v as number) })));
    if (typeof st.durationMs === 'number') timings.unshift(kvRow('Capture', h('span', { text: formatDuration(st.durationMs) })));
    const about: HTMLElement[] = [kvRow('Mode', h('span', { text: modeLabel(r.mode) })), kvRow('Quality', h('span', { text: qualityLabel(r.quality) }))];
    if (typeof st.camera === 'string') about.push(kvRow('Camera', h('span', { text: String(st.camera) })));
    if (typeof st.resolution === 'string') about.push(kvRow('Resolution', h('span', { text: String(st.resolution) })));
    const content: HTMLElement[] = [section('Scale & orientation', group(...rows)), section('Reconstruction', group(...counts))];
    const mem = memoryRows(st);
    if (mem.length) content.push(section('Memory', group(...mem)));
    if (timings.length) content.push(section('Timings', group(...timings)));
    content.push(section('Capture', group(...about)));
    openSheet({ title: 'Model info', subtitle: `${modeLabel(r.mode)} scan · ${qualityLabel(r.quality)} quality`, content, testId: 'sheet-info' });
  }

  // ---- photoreal build ----------------------------------------------------------------------------
  /**
   * Trains the Gaussian splats (engine.splat) while the model stays explorable: progress card with
   * Cancel; the wake lock is held and leaving the page asks first. A cancelled / failed run is
   * marked resumable (the engine keeps checkpoints; the same quality continues from the last one).
   */
  async function buildSplats(q: QualityPreset): Promise<void> {
    if (building || closed || !app.results.sparse) return;
    building = true;
    cancelling = false;
    app.busy = true;
    // iOS suspends a hidden page; the engine pauses its GPU work while hidden anyway
    void app.wakeLock.acquire();
    jobBar.set(-1);
    jobSub.textContent = 'Starting';
    jobCancel.disabled = false;
    renderBadges();
    const sessionId = route.sessionId;
    try {
      const res = await app.engine.splat({ quality: q }, (e: ProgressEvent) => {
        if (closed || cancelling) return;
        const v = splatProgressView(e);
        jobBar.set(v.progress);
        jobSub.textContent = v.text;
      });
      const r = app.results;
      r.splat = res;
      const timings: Record<string, number> = {};
      for (const k of ['splatPrepareMs', 'splatTrainMs', 'splatFinishMs']) if (typeof res.stats[k] === 'number') timings[k] = res.stats[k] as number;
      r.stats = { ...r.stats, ...timings, splatQuality: q };
      clearSplatResume(r.stats);
      // a splat PLY prepared before holds the previous model
      splatVersion++;
      exports.delete('ply-splat');
      void app.updateSession(sessionId, { stats: { splatGaussians: res.count, splatIterations: res.iterations, splatQuality: q } });
      if (closed) return;
      renderExports?.();
      toast(`Photoreal model ready · ${formatCount(res.count)} Gaussians`, { tone: 'good', icon: 'splats' });
      // a new model: show it (also replaces an older one in the viewer)
      void loadSplats(true);
    } catch (e) {
      const r = app.results;
      r.stats = { ...r.stats, splatInterrupted: 1, splatInterruptedQuality: q };
      if (closed) return;
      if (isCancelledError(e) || cancelling) toast('Stopped. Resume continues from the saved training.', { tone: 'info', icon: 'splats' });
      else toast(`Couldn’t build the photoreal model: ${(e as Error)?.message ?? e}`, { tone: 'bad', durationMs: 9000 });
    } finally {
      building = false;
      cancelling = false;
      // once the screen was left, leave() released them and the next screen may hold them now
      if (!closed) {
        app.busy = false;
        void app.wakeLock.release();
        renderBadges();
      }
    }
  }

  async function cancelBuild(): Promise<void> {
    if (!building || cancelling) return;
    const ok = await confirmDialog({
      title: 'Stop building the photoreal model?',
      message: 'The training so far is kept: Resume continues from it.',
      confirmLabel: 'Stop',
      cancelLabel: 'Continue',
      destructive: true,
    });
    if (!ok || !building || closed) return;
    stopBuild();
  }

  /** Cancel the running build; restart the engine if it does not stop within 8 s. */
  function stopBuild(): void {
    cancelling = true;
    jobCancel.disabled = true;
    jobSub.textContent = 'Stopping…';
    jobBar.set(-1);
    void app.engine.abortLongOperation({ graceMs: 8000, method: 'splat' });
  }

  async function goBack(): Promise<void> {
    if (building && !cancelling) {
      const ok = await confirmDialog({
        title: 'Stop building the photoreal model?',
        message: 'The training so far is kept: open the scan again and tap Resume to continue.',
        confirmLabel: 'Stop and leave',
        cancelLabel: 'Stay',
        destructive: true,
      });
      if (!ok || closed) return;
      if (building) stopBuild();
    }
    app.navigate({ type: 'BACK' });
  }

  // ---- lifecycle ---------------------------------------------------------------------------------
  async function load(): Promise<void> {
    const r = app.results;
    const need = !!route.sessionId && (app.loadedSessionId !== route.sessionId || (!r.sparse && !r.dense && !r.mesh));
    if (need) {
      app.loadedSessionId = app.loadedSessionId === route.sessionId ? null : app.loadedSessionId;
      if (!(await app.ensureSessionLoaded(route.sessionId))) {
        if (!closed) app.navigate({ type: 'BACK' });
        return;
      }
    }
    if (closed) return;
    loading.hidden = true;
    if (!app.env.webgl) {
      replaceChildren(canvasHost, h('div', { class: 'viewer-nogl' }, icon('warning', { size: 28 }), h('p', { text: 'WebGL is not available, so the model cannot be displayed. You can still export it.' })));
      applyContent(true);
      return;
    }
    try {
      const { Viewer: ViewerCtor } = await loadViewerModule();
      if (closed) return;
      viewer = new ViewerCtor(canvasHost, { background: '#0b0c10', onSplatState: (st, detail) => onSplatState(st, detail) });
      app.debug.viewer = viewer;
    } catch (e) {
      replaceChildren(canvasHost, h('div', { class: 'viewer-nogl' }, icon('warning', { size: 28 }), h('p', { text: `The 3D view could not start: ${(e as Error)?.message ?? e}` })));
    }
    applyContent(true);
  }

  return {
    el,
    enter() {
      void load();
    },
    leave() {
      closed = true;
      if (building) {
        // the training must not outlive its screen (it keeps its checkpoints)
        if (!cancelling) void app.engine.abortLongOperation({ graceMs: 8000, method: 'splat' });
        app.busy = false;
        void app.wakeLock.release();
      }
      if (app.debug.viewer === viewer) app.debug.viewer = undefined;
      viewer?.dispose();
      viewer = null;
      d.dispose();
    },
  };
}

/** Engine memory statistics (stats.memoryMB after the last step, the dense plan's peak, size caps). */
function memoryRows(st: Record<string, number | string>): HTMLElement[] {
  return memoryStatRows(st).map(([k, v]) => kvRow(k, h('span', { text: v })));
}

function timingLabel(k: string): string {
  const base = k.replace(/Ms$/, '');
  const map: Record<string, string> = {
    matching: 'Matching',
    reconstruction: 'Reconstruction',
    alignment: 'Alignment',
    depthMaps: 'Depth maps',
    fusion: 'Fusion',
    meshing: 'Meshing',
    total: 'Total',
    splatPrepare: 'Photoreal: preparation',
    splatTrain: 'Photoreal: training',
    splatFinish: 'Photoreal: saving',
  };
  return map[base] ?? base.replace(/([A-Z])/g, ' $1').replace(/^./, (c) => c.toUpperCase());
}
