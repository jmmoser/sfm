/**
 * Viewer: full-screen 3D model (points / mesh / cameras / EDL), measure tool (metres only for
 * metric models), Build sheet (dense cloud / mesh with a quality picker), Export sheet
 * (PLY points, PLY mesh, GLB, COLMAP + images ZIP: the first tap builds the file through the
 * engine and shows its size, a second tap "Share / Save" calls saveFile directly in the tap
 * handler — iOS allows navigator.share only within ~5 s of a user activation), Info sheet
 * (scale source and uncertainty, gravity / north alignment with the yaw source — the compass
 * gives MAGNETIC north —, counts, reprojection error, engine memory, timings). A densify that
 * was interrupted (stats.denseResumablePass from loadSession, or cancelled / failed here) offers
 * "Resume dense reconstruction" (HUD + Build sheet): the engine continues from its stored maps.
 */
import { saveFile } from '../../io/download';
import type { Viewer } from '../../viewer';
import { loadViewerModule } from '../viewer-loader';
import { formatDistance } from '../../viewer/helpers';
import type { ExportFormat, QualityPreset } from '../../workers/protocol';
import type { App, Results, Screen } from '../app';
import { button, chip, group, iconButton, kvRow, section, segmented, setButtonLabel, spinner, switchRow } from '../components/controls';
import { openSheet, type SheetHandle } from '../components/sheet';
import { toast } from '../components/toast';
import { Disposer, h, replaceChildren } from '../dom';
import { exportFilename, formatBytes, formatCount, formatDuration, modeLabel, qualityLabel } from '../format';
import { icon, type IconName } from '../icons';
import type { ProcessingStep, Route } from '../router';
import { builtQuality, clearDenseResume, denseResumable, denseResumeQuality, memoryStatRows, northInfo } from '../model-info';

interface DisplayState {
  points: boolean;
  mesh: boolean;
  cameras: boolean;
  edl: boolean;
  grid: boolean;
}

interface ExportState {
  status: 'idle' | 'building' | 'ready' | 'error';
  blob?: Blob;
  error?: string;
}

const EXPORTS: { format: ExportFormat; title: string; icon: IconName; ext: string; suffix: string }[] = [
  { format: 'ply-points', title: 'Point cloud (PLY)', icon: 'points', ext: 'ply', suffix: 'points' },
  { format: 'ply-mesh', title: 'Mesh (PLY)', icon: 'mesh', ext: 'ply', suffix: 'mesh' },
  { format: 'glb', title: '3D model (GLB)', icon: 'cube', ext: 'glb', suffix: '' },
  { format: 'colmap-zip', title: 'COLMAP project (ZIP)', icon: 'zip', ext: 'zip', suffix: 'colmap' },
];

export function createViewerScreen(app: App, route: Extract<Route, { screen: 'viewer' }>): Screen {
  const d = new Disposer();
  let viewer: Viewer | null = null;
  let closed = false;
  /** load() applied this scan: until then app.results may still describe the previous one. */
  let ready = false;
  let measuring = false;
  const display: DisplayState = { points: true, mesh: false, cameras: true, edl: true, grid: true };
  const exports = new Map<ExportFormat, ExportState>();

  const canvasHost = h('div', { class: 'viewer-canvas', 'data-testid': 'viewer-canvas' });
  const loading = h('div', { class: 'viewer-loading' }, spinner('lg'), h('span', { text: 'Loading model…' }));
  // Opened from Home, app.results may still hold the previously loaded scan until load() ran.
  const titleEl = h('span', { class: 'viewer-title', text: route.sessionId && app.loadedSessionId !== route.sessionId ? '' : app.results.name });
  const subEl = h('span', { class: 'viewer-sub' });
  const top = h(
    'header',
    { class: 'viewer-top' },
    iconButton('back', 'Back to scans', () => app.navigate({ type: 'BACK' }), { variant: 'glass', testId: 'viewer-back' }),
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

  const el = h('div', { class: 'viewer-screen' }, canvasHost, loading, top, resumeHud, measureHud, toolbar);

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
  function applyContent(first: boolean): void {
    const r = app.results;
    // (an empty dense cloud, from a fusion that kept nothing, shows the sparse one, as exports do)
    const dense = r.dense && r.dense.count > 0 ? r.dense : null;
    titleEl.textContent = r.name;
    const parts: string[] = [];
    if (r.mesh) parts.push(`${formatCount(r.mesh.triangleCount)} triangles`);
    if (dense) parts.push(`${formatCount(dense.count)} points`);
    else if (r.sparse) parts.push(`${formatCount(r.sparse.points.count)} points`);
    if (r.sparse) parts.push(`${r.sparse.cameras.length} photos`);
    subEl.textContent = parts.join(' · ');
    buildTool.classList.toggle('has-badge', !r.dense || !r.mesh);
    resumeHud.hidden = !denseResumable(r);
    const rq = denseResumeQuality(r);
    resumeSub.textContent = rq ? `Resume from the saved depth maps (${qualityLabel(rq)} quality).` : 'Resume from the saved depth maps.';
    if (!viewer) return;
    const metric = r.sparse?.metric;
    viewer.setWorldUp(metric?.gravityAligned ? 'z' : 'auto');
    viewer.setUnits(!!metric?.metric);
    if (dense) viewer.setPointCloud(dense);
    else if (r.sparse) viewer.setPointCloud(r.sparse.points, { pointSizePx: 3, splats: false });
    else viewer.setPointCloud(null);
    viewer.setMesh(r.mesh);
    viewer.setCameras(r.sparse ? r.sparse.cameras.map((c) => ({ pose: c.pose, intrinsics: c.intrinsics, upCam: c.upCam })) : [], false);
    if (first) {
      display.mesh = !!r.mesh;
      display.points = !r.mesh;
      display.grid = !!metric?.gravityAligned;
    }
    if (!r.mesh) display.mesh = false;
    if (!display.points && !display.mesh) display.points = true;
    viewer.setDisplay(display);
    if (first) viewer.fitToContent();
  }

  function setDisplay(patch: Partial<DisplayState>): void {
    Object.assign(display, patch);
    viewer?.setDisplay(display);
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
    if (!ready) return;
    const r = app.results;
    const rows = [
      switchRow({ label: 'Points', icon: 'points', checked: display.points, testId: 'display-points', onChange: (v) => setDisplay({ points: v }) }),
      switchRow({
        label: 'Mesh',
        icon: 'mesh',
        description: r.mesh ? undefined : 'Not built yet',
        checked: display.mesh,
        disabled: !r.mesh,
        testId: 'display-mesh',
        onChange: (v) => setDisplay({ mesh: v }),
      }),
      switchRow({ label: 'Cameras', icon: 'frustum', checked: display.cameras, testId: 'display-cameras', onChange: (v) => setDisplay({ cameras: v }) }),
      switchRow({ label: 'Eye-dome lighting', icon: 'sun', description: 'Shades point clouds for depth', checked: display.edl, testId: 'display-edl', onChange: (v) => setDisplay({ edl: v }) }),
      switchRow({ label: 'Ground grid', icon: 'target', checked: display.grid, disabled: !r.sparse?.metric.gravityAligned, onChange: (v) => setDisplay({ grid: v }) }),
    ];
    const reset = button({ label: 'Reset view', icon: 'target', variant: 'secondary', size: 'md', block: true, onClick: () => viewer?.fitToContent(true) });
    openSheet({ title: 'Display', content: [group(...rows.map((r) => r.el)), h('div', { class: 'sheet-footer' }, reset)], testId: 'sheet-display' });
  }

  function openBuild(): void {
    if (!ready) return;
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
      testId: string;
      disabled?: boolean;
    }): HTMLElement => {
      const sub = h('span', { class: 'build-sub' });
      const action = h('div', { class: 'build-action' });
      const sync = (): void => {
        const built = o.done && o.done.quality === q;
        sub.textContent = !o.done ? o.desc : built ? `${o.done.sub} · ${qualityLabel(o.done.quality)}` : o.rebuildDesc(q);
        replaceChildren(
          action,
          built
            ? chip('Built', 'good', 'check')
            : button({ label: o.done ? 'Rebuild' : 'Build', variant: 'primary', size: 'sm', testId: o.testId, disabled: o.disabled, onClick: () => build(o.steps) }),
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
                disabled: !r.sparse,
              }),
          item({
            icon: 'mesh',
            title: 'Mesh',
            done: r.mesh ? { sub: `${formatCount(r.mesh.triangleCount)} triangles`, quality: builtQuality(r, 'mesh') } : null,
            desc: r.dense ? 'A coloured surface from the dense cloud.' : 'Builds the dense cloud first, then the surface.',
            rebuildDesc: (q) => `Rebuilds the dense cloud and mesh at ${qualityLabel(q)} quality.`,
            steps: r.dense ? ['mesh'] : ['densify', 'mesh'],
            testId: 'build-mesh',
            disabled: !r.sparse,
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

  /** Re-renders the OPEN export sheet (null when closed): a build that finishes after the sheet
   *  was closed and opened again must update the new sheet, not the detached old list. */
  let renderExports: (() => void) | null = null;
  /** Until this time (or the share settles) further Save taps are ignored: a double tap would
   *  reject (InvalidStateError) and, in a Safari tab, fall back to a download on top of the share
   *  sheet. Time-limited because a share promise that never settles must not disable the button. */
  let sharingUntil = 0;

  function openExport(): void {
    if (!ready) return;
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
    const av = exportAvailability(app.results, e.format);
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
    exports.set(format, { status: 'building' });
    renderExports?.();
    try {
      const blob = await app.engine.exportModel(format);
      exports.set(format, { status: 'ready', blob });
    } catch (e) {
      exports.set(format, { status: 'error', error: (e as Error)?.message ?? String(e) });
    }
    if (!closed) renderExports?.();
  }

  function openInfo(): void {
    if (!ready) return;
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
    ready = true;
    loading.hidden = true;
    if (!app.env.webgl) {
      replaceChildren(canvasHost, h('div', { class: 'viewer-nogl' }, icon('warning', { size: 28 }), h('p', { text: 'WebGL is not available, so the model cannot be displayed. You can still export it.' })));
      applyContent(true);
      return;
    }
    try {
      const { Viewer: ViewerCtor } = await loadViewerModule();
      if (closed) return;
      viewer = new ViewerCtor(canvasHost, { background: '#0b0c10' });
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
      viewer?.dispose();
      viewer = null;
      d.dispose();
    },
  };
}

/** Whether an export format has something to export, and what (the Export sheet's sub line). */
export function exportAvailability(r: Pick<Results, 'sparse' | 'dense' | 'mesh'>, f: ExportFormat): { ok: boolean; sub: string } {
  // an empty dense cloud (fusion kept no point) is skipped like the engine's export does
  const dense = r.dense && r.dense.count > 0 ? r.dense : null;
  switch (f) {
    case 'ply-points':
      if (dense) return { ok: true, sub: `Dense cloud · ${formatCount(dense.count)} points, normals, colours` };
      if (r.sparse) return { ok: true, sub: `Sparse points · ${formatCount(r.sparse.points.count)}` };
      return { ok: false, sub: 'No points yet' };
    case 'ply-mesh':
      return r.mesh ? { ok: true, sub: `${formatCount(r.mesh.triangleCount)} triangles with vertex colours` } : { ok: false, sub: 'Build the mesh first' };
    case 'glb':
      if (r.mesh) return { ok: true, sub: 'Coloured mesh — opens in most 3D apps and AR' };
      if (dense || r.sparse) return { ok: true, sub: 'Coloured point cloud (glTF binary)' };
      return { ok: false, sub: 'Nothing to export yet' };
    case 'colmap-zip':
      return r.sparse ? { ok: true, sub: `Cameras, sparse model and ${r.sparse.cameras.length} photos` } : { ok: false, sub: 'No sparse model yet' };
  }
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
  };
  return map[base] ?? base.replace(/([A-Z])/g, ' $1').replace(/^./, (c) => c.toUpperCase());
}
