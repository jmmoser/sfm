/**
 * Capture: full-bleed camera preview, keypoint overlay (object-fit: cover mapping), tracking
 * status / counters / elapsed time / GPS accuracy, guidance banner, collapsible live 3D view
 * (≤ 2 Hz, decimated), pause / finish / torch / cancel. A toggle swaps the camera preview for the
 * live point cloud full screen, seen from the phone (FollowView: newest keyframe + gyro); the
 * preview keeps playing underneath — frames are grabbed from it.
 *
 * Pipeline: CameraCapture → FramePump (≤ 10 Hz, back-pressure: the next frame is grabbed only
 * after engine.processFrame resolved) → engine. WakeLock while capturing (acquired in the Start
 * tap). Lifecycle (docs/research/ios-camera.md §6, ios-platform.md §6.2):
 *  - page hidden ⇒ pause; on return a TAP resumes (re-acquiring the camera inside the gesture
 *    when the track did not come back — iOS reuses a camera grant without a gesture only ~1 min);
 *  - camera errors show a recovery card whose button restarts the camera from the tap;
 *  - UI rotation: the layout reflows, frames carry their screenAngle; the overlay of a frame
 *    captured before the rotation is dropped; a RAW sensor buffer (landscape bitmap in a portrait
 *    UI, upright for θ = 90 — the engine's rule) is turned onto the upright preview;
 *  - finishing can be cancelled (a stuck finishCapture is aborted, the worker restarted);
 *  - the engine's capture memory limit ('out-of-memory' event) asks "finish scan now?".
 */
import type { PumpFrame } from '../../sensors/frame-pump';
import { CameraCapture, type CameraStartResult, type CameraState } from '../../sensors/camera';
import { CompassSensor } from '../../sensors/orientation';
import { FramePump } from '../../sensors/frame-pump';
import { GeoSensor } from '../../sensors/geolocation';
import { integrateGyro, MotionSensor } from '../../sensors/motion';
import { screenAngle, type ScreenAngle } from '../../sensors/frame-orientation';
import type { CapturePermissionStatus } from '../../sensors/permissions';
import type { Viewer } from '../../viewer';
import { loadViewerModule } from '../viewer-loader';
import { ENGINE_ERRORS, isCancelledError, type FrameInput, type LiveCamera, type LiveUpdate } from '../../workers/protocol';
import type { App, Screen } from '../app';
import type { AbortOutcome, EngineErrorInfo } from '../engine-client';
import { button, iconButton, setButtonLabel, spinner } from '../components/controls';
import { confirmDialog } from '../components/dialog';
import { toast } from '../components/toast';
import { Disposer, h, replaceChildren } from '../dom';
import { defaultSessionName, formatAccuracy, formatBytes, formatClock, formatCount, guidanceInfo, trackingLabel } from '../format';
import { icon, type IconName } from '../icons';
import { frameErrorAction, isQuotaError, overlayMatchesView } from '../capture-policy';
import { FrameOrientationTracker, mapOverlay } from '../overlay-map';
import { createDebugOverlay } from '../debug-overlay';
import { coverFovDeg, FollowView, medianDistance, quarterTurned } from '../follow-view';
import { MIN_KEYFRAMES, type Route } from '../router';

type Phase = 'starting' | 'running' | 'paused' | 'interrupted' | 'error' | 'finishing' | 'closed';

interface FrameMeta {
  width: number;
  height: number;
  screenAngle: number;
  /** θ the bitmap's pixels are upright for (90 for raw sensor buffers); null = dropped by the engine. */
  upright: ScreenAngle | null;
}

const MINI_POINTS = 20_000;
const EXPANDED_POINTS = 50_000;
const MAX_CAMERAS = 80;
/** Cloud view: pose updates at most this often (ms) — the viewer renders next to the engine's GPU work. */
const FOLLOW_FRAME_MS = 30;
/** Below this much free origin quota a capture starts with a warning. */
const LOW_STORAGE_BYTES = 300e6;

export function createCaptureScreen(app: App, route: Extract<Route, { screen: 'capture' }>): Screen {
  const d = new Disposer();
  const cfg = { ...app.config };
  const sessionId = route.sessionId;
  const sessionName = defaultSessionName(cfg.mode);
  app.results.name = sessionName;
  app.results.mode = cfg.mode;
  app.results.quality = cfg.quality;

  const camera = new CameraCapture();
  /** The "VideoFrame grab" setting (where VideoFrame exists): no main-thread copy per grab. */
  const frameGrab: 'bitmap' | 'videoFrame' = cfg.videoFrameGrab && typeof (globalThis as { VideoFrame?: unknown }).VideoFrame === 'function' ? 'videoFrame' : 'bitmap';
  const motion = MotionSensor.isSupported() ? new MotionSensor() : null;
  const compass = new CompassSensor();
  const geo = cfg.useGps && GeoSensor.isSupported() ? new GeoSensor() : null;
  let pump: FramePump | null = null;
  let camRes: CameraStartResult | null = null;
  let phase: Phase = 'starting';
  let closed = false;
  let torchOn = false;
  let sessionCreated: Promise<unknown> = Promise.resolve();

  // Live state.
  let last: LiveUpdate | null = null;
  let numKeyframes = 0;
  let elapsedMs = 0;
  let runningSince: number | null = null;
  let thumbDone = false;
  let latestSparse: { positions: Float32Array; colors: Uint8Array } | null = null;
  let latestCameras: LiveCamera[] | null = null;
  /** R_cam_device of the poses in latestCameras (engine camera frame) and the engine's gyro bias. */
  let latestRcd: Float64Array | null = null;
  let gyroBias: [number, number, number] | null = null;
  /** Bitmap size / UI rotation of the frame being processed and the θ its pixels are upright for. */
  let inFlight: FrameMeta | null = null;
  /** Raw sensor buffers vs upright frames — the engine's rule, to turn overlays onto the preview. */
  const orientation = new FrameOrientationTracker();
  let frameErrors = 0;

  // ---- DOM ---------------------------------------------------------------------------------------
  const video = h('video', { class: 'capture-video', playsinline: true, muted: true, autoplay: true, 'aria-label': 'Camera preview' });
  const overlay = h('canvas', { class: 'capture-overlay', 'aria-hidden': 'true' });
  const flash = h('div', { class: 'capture-flash', 'aria-hidden': 'true' });
  const dot = h('span', { class: 'status-dot' });
  const trackLabel = h('span', { class: 'status-label', text: 'Starting' });
  const timeLabel = h('span', { class: 'status-time', text: '0:00' });
  const statusPill = h('div', { class: 'status-pill', role: 'status', 'data-state': 'initializing', 'data-testid': 'tracking' }, dot, trackLabel, h('span', { class: 'status-sep' }), timeLabel);
  const gpsChip = h('div', { class: 'gps-chip', hidden: true, 'data-testid': 'gps' }, icon('location', { size: 14 }), h('span', { text: 'GPS…' }));
  // Shown after the engine's capture memory cap: tracking continues, no new keyframes.
  const memChip = h('div', { class: 'gps-chip warn capture-mem-chip', hidden: true, role: 'status', 'data-testid': 'memory-limited' }, icon('cpu', { size: 14 }), h('span', { text: 'Memory full · tracking only' }));
  const kfCount = h('strong', { text: '0', 'data-testid': 'kf-count' });
  const regCount = h('strong', { text: '0' });
  const ptCount = h('strong', { text: '0' });
  const stats = h(
    'div',
    { class: 'capture-stats', 'aria-live': 'off' },
    h('span', { class: 'stat' }, kfCount, h('span', { text: 'keyframes' })),
    h('span', { class: 'stat' }, regCount, h('span', { text: 'registered' })),
    h('span', { class: 'stat' }, ptCount, h('span', { text: 'points' })),
  );
  const closeBtn = iconButton('close', 'Cancel scan', () => void cancel(), { variant: 'glass', testId: 'capture-cancel' });
  const viewBtn = iconButton('cube', 'Point cloud view', () => setCloudView(!cloudView), { variant: 'glass', testId: 'cloud-toggle' });
  viewBtn.setAttribute('aria-pressed', 'false');
  // no WebGL: no live 3D view to show instead of the camera
  viewBtn.hidden = !app.env.webgl;
  const top = h(
    'div',
    { class: 'capture-top' },
    h('div', { class: 'capture-top-row' }, closeBtn, statusPill, h('div', { class: 'capture-top-right' }, gpsChip, viewBtn)),
    stats,
    memChip,
  );
  const guidanceTitle = h('span', { class: 'guidance-title' });
  const guidanceDetail = h('span', { class: 'guidance-detail' });
  const guidance = h('div', { class: 'guidance', role: 'status', 'aria-live': 'polite', hidden: true, 'data-testid': 'guidance' }, h('span', { class: 'guidance-icon' }), h('span', { class: 'guidance-text' }, guidanceTitle, guidanceDetail));

  const miniHost = h('div', { class: 'mini-view-canvas' });
  const miniEmptyLabel = h('span', { text: 'Live 3D' });
  const miniEmpty = h('div', { class: 'mini-view-empty' }, icon('cube', { size: 22 }), miniEmptyLabel);
  const miniCollapse = iconButton('collapse', 'Collapse 3D view', (e) => {
    e.stopPropagation();
    setExpanded(false);
  }, { variant: 'glass', className: 'mini-collapse' });
  const mini = h('div', { class: 'mini-view', role: 'button', tabindex: '0', 'aria-label': 'Expand live 3D view', 'data-testid': 'mini-view' }, miniHost, miniEmpty, miniCollapse);

  const torchBtn = iconButton('torchOff', 'Torch', () => void toggleTorch(), { variant: 'glass', className: 'ctl-side', testId: 'torch' });
  torchBtn.setAttribute('aria-pressed', 'false');
  const pauseBtn = h('button', { type: 'button', class: 'ctl-main', 'aria-label': 'Pause capture', 'data-testid': 'pause' }, h('span', { class: 'ctl-main-ring' }), h('span', { class: 'ctl-main-core' }, icon('pause', { size: 28 })));
  pauseBtn.addEventListener('click', () => (phase === 'paused' ? resume() : pause('user')));
  const finishBtn = button({ label: 'Finish', icon: 'check', variant: 'primary', size: 'md', className: 'ctl-finish', testId: 'finish', onClick: () => void finish() });
  finishBtn.disabled = true;
  const finishHint = h('div', { class: 'finish-hint', 'aria-live': 'polite', text: `Capture ${MIN_KEYFRAMES} keyframes to finish` });
  const controls = h('div', { class: 'capture-controls' }, finishHint, h('div', { class: 'capture-controls-row' }, torchBtn, pauseBtn, finishBtn));

  const debugOverlay = cfg.debugOverlay ? createDebugOverlay() : null;
  const card = h('div', { class: 'capture-card-host' });
  const pausedLabel = h('div', { class: 'paused-label', hidden: true, text: 'Paused' });

  const el = h(
    'div',
    { class: 'capture', 'data-phase': 'starting' },
    video,
    overlay,
    flash,
    h('div', { class: 'capture-shade-top', 'aria-hidden': 'true' }),
    h('div', { class: 'capture-shade-bottom', 'aria-hidden': 'true' }),
    top,
    guidance,
    pausedLabel,
    mini,
    debugOverlay?.el ?? null,
    controls,
    card,
  );

  // ---- UI helpers --------------------------------------------------------------------------------
  function setPhase(p: Phase): void {
    phase = p;
    el.dataset.phase = p;
    const paused = p === 'paused';
    pausedLabel.hidden = !paused;
    if (p !== 'running') {
      guidance.hidden = true;
      lastGuidance = '';
    }
    pauseBtn.setAttribute('aria-label', paused ? 'Resume capture' : 'Pause capture');
    pauseBtn.querySelector('.ctl-main-core')!.replaceChildren(icon(paused ? 'play' : 'pause', { size: 28 }));
    pauseBtn.disabled = !(p === 'running' || p === 'paused');
    // The pill shows the live tracking state only while frames flow.
    if (paused || p === 'interrupted') {
      statusPill.dataset.state = 'paused';
      trackLabel.textContent = 'Paused';
    } else if (p === 'running' && last) {
      statusPill.dataset.state = last.tracking;
      trackLabel.textContent = trackingLabel(last.tracking);
    }
    if (p === 'running') {
      if (runningSince === null) runningSince = performance.now();
    } else if (runningSince !== null) {
      elapsedMs += performance.now() - runningSince;
      runningSince = null;
    }
    updateFinish();
  }

  function elapsed(): number {
    return elapsedMs + (runningSince !== null ? performance.now() - runningSince : 0);
  }

  function updateFinish(): void {
    const enough = numKeyframes >= MIN_KEYFRAMES;
    finishBtn.disabled = !enough || !(phase === 'running' || phase === 'paused');
    finishHint.hidden = enough || phase === 'starting' || phase === 'error';
    if (!enough) finishHint.textContent = `${numKeyframes} / ${MIN_KEYFRAMES} keyframes — keep moving`;
  }

  function showCard(content: HTMLElement | null): void {
    replaceChildren(card, content);
    el.classList.toggle('has-card', !!content);
  }

  function infoCard(o: { icon?: IconName; spinner?: boolean; title: string; text?: string; actions?: HTMLElement[]; tone?: 'neutral' | 'bad'; testId?: string }): HTMLElement {
    return h(
      'div',
      { class: `capture-card${o.tone === 'bad' ? ' capture-card-bad' : ''}`, role: o.tone === 'bad' ? 'alert' : 'status', 'data-testid': o.testId },
      o.spinner ? spinner('lg') : o.icon ? h('span', { class: 'capture-card-icon' }, icon(o.icon, { size: 30 })) : null,
      h('h2', { class: 'capture-card-title', text: o.title }),
      o.text ? h('p', { class: 'capture-card-text', text: o.text }) : null,
      o.actions?.length ? h('div', { class: 'capture-card-actions' }, ...o.actions) : null,
    );
  }

  function errorCard(title: string, text: string, retry: (() => void) | null, retryLabel = 'Try again'): void {
    setPhase('error');
    const actions: HTMLElement[] = [];
    if (retry) actions.push(button({ label: retryLabel, variant: 'primary', size: 'lg', block: true, onClick: retry, testId: 'capture-retry' }));
    actions.push(button({ label: 'Back to home', variant: 'secondary', size: 'lg', block: true, onClick: () => void abandon(), testId: 'capture-back' }));
    showCard(infoCard({ icon: 'warning', title, text: capitalize(text), actions, tone: 'bad', testId: 'capture-error' }));
  }

  // ---- overlay -----------------------------------------------------------------------------------
  let overlayClearTimer: ReturnType<typeof setTimeout> | null = null;
  function sizeOverlay(): { w: number; h: number; dpr: number } {
    const r = el.getBoundingClientRect();
    // ≤ 2×: a 3× full-screen canvas is ~12 MB cleared and redrawn 10 times a second next to the
    // engine's GPU work; 2.2 px dots look the same at 2×.
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(r.width * dpr));
    const hh = Math.max(1, Math.round(r.height * dpr));
    if (overlay.width !== w || overlay.height !== hh) {
      overlay.width = w;
      overlay.height = hh;
    }
    return { w: r.width, h: r.height, dpr };
  }

  function drawOverlay(pts: Float32Array | undefined, meta: FrameMeta): void {
    const ctx = overlay.getContext('2d');
    if (!ctx) return;
    const { w, h: hh, dpr } = sizeOverlay();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    // A frame from before a UI rotation does not match the rotated preview: drop its overlay.
    const display = screenAngle();
    if (!pts || pts.length < 2 || meta.upright === null || !overlayMatchesView(meta.screenAngle, display)) return;
    // The overlay is relative to the bitmap AS SUBMITTED; a raw sensor buffer (landscape bitmap in
    // a portrait UI, upright for θ = 90) needs a quarter turn onto the upright preview.
    const quarterTurns = FrameOrientationTracker.overlayTurns(meta.upright, display);
    const xy = mapOverlay(pts, meta.width, meta.height, w, hh, { fit: 'cover', quarterTurns });
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = 'rgba(54, 224, 166, 0.95)';
    ctx.strokeStyle = 'rgba(4, 18, 12, 0.55)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < xy.length; i += 2) {
      ctx.moveTo(xy[i] + 2.2, xy[i + 1]);
      ctx.arc(xy[i], xy[i + 1], 2.2, 0, Math.PI * 2);
    }
    ctx.fill();
    ctx.stroke();
    if (overlayClearTimer) clearTimeout(overlayClearTimer);
    overlayClearTimer = setTimeout(() => {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, overlay.width, overlay.height);
    }, 900);
  }

  function clearOverlay(): void {
    const ctx = overlay.getContext('2d');
    ctx?.clearRect(0, 0, overlay.width, overlay.height);
  }

  // ---- mini 3D view ------------------------------------------------------------------------------
  let viewer: Viewer | null = null;
  let viewerFailed = false;
  let expanded = false;
  let lastViewerUpdate = 0;
  let viewerTimer: ReturnType<typeof setTimeout> | null = null;
  let viewerDirty = false;
  /** The user orbited the expanded view: stop re-framing it until it is collapsed. */
  let userMovedView = false;
  miniHost.addEventListener('pointerdown', () => {
    if (expanded) userMovedView = true;
  });

  function setExpanded(v: boolean): void {
    expanded = v;
    userMovedView = false;
    if (v) clearOverlay();
    mini.classList.toggle('expanded', v);
    // Collapsed it is a button; expanded it is a region holding the (separate) collapse button —
    // never a button inside a button for VoiceOver.
    mini.setAttribute('role', v ? 'region' : 'button');
    mini.tabIndex = v ? -1 : 0;
    mini.setAttribute('aria-label', v ? 'Live 3D view' : 'Expand live 3D view');
    if (v) miniCollapse.focus({ preventScroll: true });
    else if (mini.contains(document.activeElement)) mini.focus({ preventScroll: true });
    el.classList.toggle('mini-expanded', v);
    viewerDirty = true;
    scheduleViewer(true);
  }
  mini.addEventListener('click', () => {
    if (!expanded && !cloudView) setExpanded(true);
  });
  mini.addEventListener('keydown', (e: KeyboardEvent) => {
    if ((e.key === 'Enter' || e.key === ' ') && !expanded && !cloudView) {
      e.preventDefault();
      setExpanded(true);
    }
  });

  let ViewerCtor: typeof Viewer | null = null;
  let viewerLoading = false;
  function ensureViewer(): Viewer | null {
    if (viewer || viewerFailed || !app.env.webgl) return viewer;
    if (!ViewerCtor) {
      // three.js is a separate chunk (usually prefetched already): load, then retry the update.
      if (!viewerLoading) {
        viewerLoading = true;
        loadViewerModule().then(
          (m) => {
            ViewerCtor = m.Viewer;
            if (!closed) scheduleViewer(true);
          },
          (e) => {
            onViewerFailed(e);
          },
        );
      }
      return null;
    }
    try {
      // autoFit 'off' + an unanimated fit per update: 'follow' animates every refit, which keeps
      // WebGL rendering continuously next to the camera preview and the engine's GPU work.
      viewer = new ViewerCtor(miniHost, { background: 'transparent', autoFit: 'off', maxPixelRatio: 2, fovDeg: 55 });
      viewer.setWorldUp('auto');
      viewer.setDisplay({ grid: false, edl: false, cameras: !cloudView, sparsePointSizePx: cloudView ? 3 : 2.5 });
    } catch (e) {
      onViewerFailed(e);
    }
    return viewer;
  }

  /** The live 3D view cannot be shown: back to the camera preview (not saved), no toggle. */
  function onViewerFailed(e: unknown): void {
    viewerFailed = true;
    console.warn('[capture] live 3D view unavailable', e);
    viewBtn.hidden = true;
    if (cloudView) {
      setCloudView(false, false);
      toast('The 3D view is not available on this device — showing the camera.', { tone: 'warn' });
    }
  }

  function scheduleViewer(now = false): void {
    if (closed) return;
    const due = lastViewerUpdate + 500 - performance.now(); // ≤ 2 Hz
    if (viewerTimer) return;
    viewerTimer = setTimeout(
      () => {
        viewerTimer = null;
        updateViewer();
      },
      now ? 0 : Math.max(0, due),
    );
  }

  function updateViewer(): void {
    if (closed || !viewerDirty || !latestSparse) return;
    const v = ensureViewer();
    if (!v) return;
    viewerDirty = false;
    lastViewerUpdate = performance.now();
    const max = expanded || cloudView ? EXPANDED_POINTS : MINI_POINTS;
    const { positions, colors } = decimate(latestSparse.positions, latestSparse.colors, max);
    v.setSparsePoints(positions, colors);
    if (latestCameras?.length) v.setCameras(decimateCameras(latestCameras, MAX_CAMERAS), true);
    if (cloudView) startFollow();
    else if (!userMovedView) v.fitToContent(false);
    miniEmpty.hidden = true;
  }

  // ---- cloud view (point cloud instead of the camera preview) ------------------------------------
  let cloudView = false;
  const follow = new FollowView(gyroRotation);
  let followRaf = 0;
  let lastFollow = 0;
  /** Pose / field of view last given to the viewer (unchanged ⇒ no render). */
  let lastViewKey = '';

  function setCloudView(v: boolean, save = true): void {
    cloudView = v;
    if (save) app.saveConfig({ cloudView: v });
    if (v) {
      if (expanded) setExpanded(false);
      clearOverlay();
    }
    el.classList.toggle('cloud-view', v);
    viewBtn.replaceChildren(icon(v ? 'camera' : 'cube', { size: 22 }));
    viewBtn.setAttribute('aria-pressed', String(v));
    mini.setAttribute('role', v ? 'img' : 'button');
    mini.tabIndex = v ? -1 : 0;
    mini.setAttribute('aria-label', v ? 'Live point cloud, seen from the camera' : 'Expand live 3D view');
    miniEmptyLabel.textContent = v ? 'The point cloud appears once tracking starts' : 'Live 3D';
    viewer?.setDisplay({ cameras: !v, sparsePointSizePx: v ? 3 : 2.5 });
    if (v) {
      follow.reset();
      lastViewKey = '';
      startFollow();
    } else {
      stopFollow();
      viewer?.setViewPose(null);
      if (viewer) viewer.fitToContent(false);
    }
    viewerDirty = true;
    scheduleViewer(true);
  }

  function startFollow(): void {
    if (!followRaf && cloudView && !closed && viewer) followRaf = requestAnimationFrame(followFrame);
  }

  function stopFollow(): void {
    if (followRaf) cancelAnimationFrame(followRaf);
    followRaf = 0;
  }

  /** Put the viewer camera where the phone is now (rAF, ≤ 1000 / FOLLOW_FRAME_MS Hz). */
  function followFrame(now: number): void {
    followRaf = 0;
    if (closed || !cloudView || !viewer) return;
    followRaf = requestAnimationFrame(followFrame);
    if (now - lastFollow < FOLLOW_FRAME_MS || !follow.hasAnchor) return;
    lastFollow = now;
    const theta = screenAngle();
    // the last IMU sample (it stays put once the sensors stop: finishing ⇒ a still view)
    const pose = follow.pose(now, motion ? motion.lastSampleTime : NaN, theta);
    if (!pose) return;
    const intr = latestCameras?.[latestCameras.length - 1]?.intrinsics;
    const fov = intr ? coverFovDeg(intr, quarterTurned(latestRcd, theta), miniHost.clientWidth, miniHost.clientHeight) : NaN;
    const key = `${Array.from(pose.R, (x) => x.toFixed(6))}|${Array.from(pose.t, (x) => x.toPrecision(7))}|${fov.toFixed(3)}`;
    if (key === lastViewKey) return;
    lastViewKey = key;
    viewer.setViewPose(pose, Number.isFinite(fov) ? fov : undefined);
  }

  /** Device rotation R_{d(t1)←d(t0)} from the gyro (bias-corrected as the engine does); null when the IMU does not cover it. */
  function gyroRotation(t0: number, t1: number): Float64Array | null {
    if (!motion) return null;
    const lo = Math.min(t0, t1),
      hi = Math.max(t0, t1);
    const samples = motion.samplesBetween(lo, hi, { pad: true });
    if (!samples.length || samples[0].t > lo + 100 || samples[samples.length - 1].t < hi - 100) return null;
    const b = gyroBias;
    const s = b ? samples.map((x) => ({ ...x, gyro: [x.gyro[0] - b[0], x.gyro[1] - b[1], x.gyro[2] - b[2]] as [number, number, number] })) : samples;
    return integrateGyro(s, t0, t1);
  }

  // ---- live updates ------------------------------------------------------------------------------
  let lastGuidance = '';
  function onLive(u: LiveUpdate): void {
    if (closed) return;
    const meta = inFlight;
    last = u;
    debugOverlay?.onLive(u);
    if (pump) debugOverlay?.setGrabMs(pump.stats.grabMs);
    const prevKf = numKeyframes;
    numKeyframes = u.numKeyframes;
    kfCount.textContent = String(u.numKeyframes);
    regCount.textContent = String(u.numRegistered);
    ptCount.textContent = formatCount(u.numPoints);
    if (phase === 'running') {
      statusPill.dataset.state = u.tracking;
      trackLabel.textContent = trackingLabel(u.tracking);
    }
    const g = guidanceInfo(u.guidance, cfg.mode);
    const gkey = `${u.guidance}`;
    if (gkey !== lastGuidance && phase === 'running') {
      lastGuidance = gkey;
      guidance.hidden = !g.title;
      guidance.dataset.tone = g.tone;
      guidanceTitle.textContent = g.title;
      guidanceDetail.textContent = g.detail;
      guidance.querySelector('.guidance-icon')!.replaceChildren(icon(g.tone === 'good' ? 'check' : g.tone === 'bad' ? 'warning' : g.tone === 'info' ? 'info' : 'warning', { size: 18 }));
    }
    const live = phase === 'running';
    if (meta && u.overlay && live && !expanded && !cloudView) drawOverlay(u.overlay, meta);
    if (live && u.decision === 'keyframe' && u.numKeyframes > prevKf) {
      flash.classList.remove('pulse');
      void flash.offsetWidth; // restart the animation
      flash.classList.add('pulse');
      if (!thumbDone && u.numKeyframes >= 1) {
        thumbDone = true;
        void saveThumbnail();
      }
    }
    if (u.sparse) {
      latestSparse = u.sparse;
      viewerDirty = true;
    }
    if (u.cameras) {
      latestCameras = u.cameras;
      latestRcd = u.R_cam_device ? Float64Array.from(u.R_cam_device) : null;
      if (u.gyroBias !== undefined) gyroBias = u.gyroBias;
      const newest = u.cameras[u.cameras.length - 1];
      // a centre jump beyond half the scene depth is a re-initialised model (shown at once)
      const depth = newest && latestSparse ? medianDistance(latestSparse.positions, cameraCentre(newest.pose)) : NaN;
      follow.setAnchor(newest ? { pose: newest.pose, t: newest.t } : null, latestRcd, Number.isFinite(depth) ? 0.5 * depth : Infinity);
      viewerDirty = true;
    }
    if (viewerDirty) scheduleViewer();
    // The engine marks every frame after its capture memory cap (in case the event was missed).
    if (u.timings?.memoryLimited) onMemoryLimit();
    updateFinish();
  }

  // ---- engine memory limit -----------------------------------------------------------------------
  let memoryLimited = false;
  function onEngineError(e: EngineErrorInfo): void {
    if (closed || e.fatal) return;
    if (e.reason === 'out-of-memory' || e.error?.name === 'CaptureMemoryLimitError') onMemoryLimit();
  }

  /**
   * The engine stopped adding keyframes (capture state at its memory budget; tracking goes on).
   * Ask once — like a full storage: pause and offer to finish now, or to continue tracking only.
   */
  function onMemoryLimit(): void {
    if (memoryLimited || closed) return;
    memoryLimited = true;
    memChip.hidden = false;
    el.classList.add('memory-limited');
    if (phase !== 'running' && phase !== 'paused') return;
    pause('user');
    const enough = numKeyframes >= MIN_KEYFRAMES;
    showCard(
      infoCard({
        icon: 'cpu',
        tone: 'bad',
        title: 'Memory limit reached — finish scan now?',
        text: enough
          ? `This device cannot hold more keyframes for this scan. Finish now to reconstruct the ${numKeyframes} keyframes captured so far, or continue with tracking only — no new keyframes are added.`
          : 'This device cannot hold more keyframes for this scan, and there are too few to reconstruct. Continue with tracking only, or start again with a lower quality.',
        testId: 'capture-memory-limit',
        actions: [
          ...(enough ? [button({ label: 'Finish scan', icon: 'check', variant: 'primary', size: 'lg', block: true, testId: 'memory-finish', onClick: () => void finish() })] : []),
          button({ label: 'Continue (tracking only)', variant: enough ? 'secondary' : 'primary', size: 'lg', block: true, testId: 'memory-continue', onClick: () => resume() }),
          ...(enough ? [] : [button({ label: 'Back to home', variant: 'secondary', size: 'lg', block: true, onClick: () => void abandon() })]),
        ],
      }),
    );
  }

  async function saveThumbnail(): Promise<void> {
    if (!sessionId) return;
    const blob = await videoThumbnail(video);
    if (!blob || closed) return;
    await sessionCreated;
    await app.updateSession(sessionId, { thumbnail: blob });
  }

  // ---- frames ------------------------------------------------------------------------------------
  async function consume(f: PumpFrame): Promise<void> {
    if (phase !== 'running' || closed) {
      f.bitmap.close();
      return;
    }
    const input: FrameInput = {
      bitmap: f.bitmap,
      t: f.t,
      screenAngle: f.screenAngle,
      imu: f.imu,
      geo: f.geo,
      // platform headings are MAGNETIC until the compass knows the declination (setLocationFromFix)
      heading: f.heading ? { heading: f.heading.heading, accuracy: f.heading.accuracy, north: f.heading.north ?? 'magnetic', declinationDeg: f.heading.declinationDeg ?? null } : null,
      index: f.index,
      segment: f.segment,
      gravityDevice: f.gravityDevice,
      attitude: f.attitude ? { t: f.attitude.t, R_world_device: f.attitude.R_world_device, absolute: f.attitude.absolute, yawAccuracyDeg: f.attitude.yawAccuracyDeg } : null,
      imuCalibrationVersion: f.imuCalibrationVersion,
      timeSource: f.timeSource,
      rawBuffer: f.rawBuffer,
      maxSide: f.maxSide,
    };
    const upright = orientation.observe({ width: f.width, height: f.height, screenAngle: f.screenAngle, t: f.t, raw: f.rawBuffer });
    inFlight = { width: f.width, height: f.height, screenAngle: f.screenAngle, upright };
    try {
      await app.engine.processFrame(input);
      frameErrors = 0;
    } catch (e) {
      onFrameError(e);
    } finally {
      inFlight = null;
    }
  }

  function onFrameError(e: unknown): void {
    if (closed) return;
    const engineState = app.engine.status.state;
    switch (frameErrorAction(e, engineState)) {
      case 'ignore':
        return;
      case 'unavailable': {
        if (phase === 'error') return;
        pump?.stop();
        clearOverlay();
        const dead = engineState === 'dead' || engineState === 'disposed';
        errorCard(
          dead ? 'Processing engine stopped' : 'Processing engine unavailable',
          dead
            ? `It failed repeatedly (${app.engine.status.reason ?? (e as Error)?.message ?? 'unknown error'}). ${numKeyframes >= MIN_KEYFRAMES ? 'The photos captured so far are kept — reload the app to process them.' : 'Reload the app and try again.'}`
            : `${(e as Error).message}. Open the app with ?mock=1 to try the demo engine.`,
          null,
        );
        return;
      }
      case 'storage': {
        if (phase !== 'running') return;
        pause('user');
        const enough = numKeyframes >= MIN_KEYFRAMES;
        const full = isQuotaError(e);
        showCard(
          infoCard({
            icon: 'storage',
            tone: 'bad',
            title: full ? 'Storage is full' : 'Could not save photos',
            text: full
              ? enough
                ? `New photos cannot be saved. Finish now to reconstruct the ${numKeyframes} keyframes captured so far, or free up space and resume.`
                : 'New photos cannot be saved. Free up space on this device (or delete old scans), then resume.'
              : `Saving to on-device storage failed (${(e as Error)?.message ?? 'unknown error'}). ${
                  enough ? `Finish now to reconstruct the ${numKeyframes} keyframes captured so far, or resume to try again.` : 'Resume to try again.'
                }`,
            testId: full ? 'capture-storage-full' : 'capture-storage-error',
            actions: [
              ...(enough ? [button({ label: 'Finish scan', icon: 'check', variant: 'primary', size: 'lg', block: true, onClick: () => void finish() })] : []),
              button({ label: 'Resume', variant: enough ? 'secondary' : 'primary', size: 'lg', block: true, onClick: () => resume() }),
              button({ label: 'Back to home', variant: 'secondary', size: 'lg', block: true, onClick: () => void abandon() }),
            ],
          }),
        );
        return;
      }
      case 'restarted':
        toast('The engine restarted — continuing the scan.', { tone: 'warn' });
        return;
      case 'transient':
        frameErrors++;
        console.warn('[capture] frame failed', e);
        if (frameErrors === 3) toast(`Frames are failing: ${(e as Error)?.message ?? e}`, { tone: 'warn' });
        return;
    }
  }

  // ---- lifecycle ---------------------------------------------------------------------------------
  async function start(): Promise<void> {
    setPhase('starting');
    showCard(infoCard({ spinner: true, title: 'Starting camera…', text: 'Allow camera and motion access when asked.' }));
    const perms = app.takePermissions();
    if (!perms) {
      // No Start tap (e.g. reload / retry): ask from a tap now.
      showPermissionTap();
      return;
    }
    const st = await perms;
    if (closed) return;
    await afterPermissions(st);
  }

  function showPermissionTap(): void {
    setPhase('error');
    showCard(
      infoCard({
        icon: 'camera',
        title: 'Camera and motion access',
        text: 'Scanning uses the camera, plus the motion sensors for scale and gravity.',
        actions: [
          button({
            label: 'Allow and start',
            variant: 'primary',
            size: 'lg',
            block: true,
            testId: 'allow-start',
            onClick: () => {
              const p = app.requestCapturePermissions(); // synchronous inside the tap
              app.takePermissions();
              setPhase('starting');
              showCard(infoCard({ spinner: true, title: 'Starting camera…' }));
              void p.then((st) => (closed ? undefined : afterPermissions(st)));
            },
          }),
          button({ label: 'Back to home', variant: 'secondary', size: 'lg', block: true, onClick: () => void abandon() }),
        ],
      }),
    );
  }

  async function afterPermissions(st: CapturePermissionStatus): Promise<void> {
    if (st.camera !== 'granted') {
      const denied = st.camera === 'denied';
      errorCard(
        denied ? 'Camera access is off' : 'Camera not available',
        denied
          ? 'Allow camera access for this website: Settings › Apps › Safari › Camera (or tap “aA” › Website Settings in Safari), then try again.'
          : `The camera could not be opened (${st.errors.join(', ') || st.camera}). Close other apps that use the camera and try again.`,
        () => showPermissionTap(),
      );
      return;
    }
    if (st.motion === 'denied' || st.motion === 'needs-gesture') {
      toast('Motion access is off — scale and gravity will be less accurate. To allow it, close and reopen the app, then tap Start.', { tone: 'warn', durationMs: 7000 });
    }
    const motionOk = st.motion === 'granted';
    if (motion && motionOk) motion.start();
    if (motionOk) compass.start();
    await openCamera();
  }

  async function openCamera(): Promise<void> {
    if (closed) return;
    setPhase('starting');
    showCard(infoCard({ spinner: true, title: 'Starting camera…' }));
    try {
      camRes = await camera.start(video, { quality: cfg.quality });
    } catch (e) {
      if (closed) return;
      const n = (e as { name?: string })?.name;
      errorCard(
        n === 'NotAllowedError' ? 'Camera access is off' : 'Camera could not start',
        n === 'NotAllowedError'
          ? 'Allow camera access for this website in Settings, then try again.'
          : n === 'NotReadableError'
            ? 'The camera is busy. Close other apps that use the camera, then try again.'
            : `${(e as Error)?.message ?? e}`,
        () => void openCamera(), // a tap: getUserMedia inside the gesture
      );
      return;
    }
    if (closed) {
      camera.stop();
      return;
    }
    d.add(camera.onStateChange(onCameraState));
    debugOverlay?.setCamera({ width: camRes.width, height: camRes.height, frameRate: camRes.frameRate, label: camRes.label, frameMaxSide: camRes.frameMaxSide, grab: frameGrab });
    torchBtn.disabled = !camRes.torchAvailable;
    if (cfg.torch && camRes.torchAvailable) void toggleTorch(true);
    // Lock white balance after auto-WB settled (iOS supports manual WB only).
    d.timeout(() => void camera.lockWhiteBalance().catch(() => false), 1200);
    if (camRes.lensMaySwitch) toast('This camera may switch lenses automatically — accuracy can suffer.', { tone: 'warn' });
    if (geo) {
      geo.start(); // location prompt (after camera, never blocking it)
      // true north: WMM declination at the first usable fix (CompassSensor.setLocationFromFix)
      d.add(
        geo.onFix((fix) => {
          compass.setLocationFromFix(fix);
          updateGps();
        }),
      );
      d.interval(updateGps, 2000);
      gpsChip.hidden = false;
    }
    if (sessionId) {
      const created = app.createSession(sessionId, { name: sessionName, mode: cfg.mode, quality: cfg.quality });
      sessionCreated = created;
      const [res, available] = await Promise.all([created, app.storageAvailable()]);
      if (closed) return;
      if (res.status === 'quota') {
        // Every keyframe would fail to save: do not start a scan that cannot be kept.
        errorCard('Storage is full', 'Free up space on this device (Settings › General › iPhone Storage) or delete old scans, then try again.', null);
        return;
      }
      if (res.status === 'error') toast('This scan cannot be saved on this device (storage error).', { tone: 'warn' });
      else if (available !== null && available < LOW_STORAGE_BYTES) {
        toast(`Only ${formatBytes(available)} of storage left — a long scan may not fit.`, { tone: 'warn', durationMs: 6000 });
      }
    }
    await initEngine();
  }

  async function initEngine(): Promise<void> {
    showCard(infoCard({ spinner: true, title: 'Preparing 3D engine…', text: 'Compiling GPU programs. This takes a moment the first time.' }));
    try {
      await app.engine.init({
        mode: cfg.mode,
        quality: cfg.quality,
        sessionId: sessionId ?? undefined,
        persist: !!sessionId,
        preferGPU: true,
        hfovPriorDeg: camRes?.hfovPriorDeg,
      });
      app.loadedSessionId = sessionId;
    } catch (e) {
      if (closed) return;
      const msg = (e as Error)?.message ?? String(e);
      errorCard(
        'Processing engine unavailable',
        (e as { name?: string })?.name === ENGINE_ERRORS.notImplemented ? `${msg}. Open the app with ?mock=1 to try the demo engine.` : msg,
        () => void initEngine(),
      );
      return;
    }
    if (closed) return;
    showCard(null);
    pump = new FramePump({
      camera,
      consumer: consume,
      motion: motion?.running ? motion : null,
      geo,
      compass: compass.running ? compass : null,
      targetHz: 10,
      grab: frameGrab,
      // raw sensor-orientation VideoFrames are verified on iOS WebKit only (Diagnostics probe)
      videoFrameOrientation: camRes?.apple ? 'raw' : 'upright',
      onError: (e) => console.warn('[capture] pump error', e),
    });
    pump.start();
    setPhase('running');
    guidance.hidden = false;
    guidanceTitle.textContent = guidanceInfo('initializing', cfg.mode).title;
    guidanceDetail.textContent = guidanceInfo('initializing', cfg.mode).detail;
    guidance.dataset.tone = 'info';
    guidance.querySelector('.guidance-icon')!.replaceChildren(icon('info', { size: 18 }));
  }

  function onCameraState(s: CameraState, detail?: string): void {
    if (closed || phase === 'finishing') return;
    if (s === 'error') {
      pump?.stop();
      errorCard('The camera stopped', `Tap to reconnect the camera${detail ? ` (${detail})` : ''}.`, () => void reconnect(), 'Reconnect camera');
    } else if (s === 'running' && phase === 'error' && pump) {
      showCard(null);
      pump.start();
      setPhase('running');
    }
  }

  /** From a tap: re-acquire the camera (inside the gesture) if needed, then resume. */
  async function reconnect(): Promise<void> {
    showCard(infoCard({ spinner: true, title: 'Reconnecting camera…' }));
    // CameraCapture may already be re-opening the camera by itself (after the page became
    // visible): restart() would refuse (false) and show a bogus error — wait for that instead.
    const busy = camera.state === 'restarting' || camera.state === 'starting';
    const ok = camera.state === 'running' ? true : busy ? await settledCamera(camera, 8000) : await camera.restart('manual');
    if (closed) return;
    if (!ok) {
      errorCard('Camera could not restart', 'Close other apps that use the camera, then try again.', () => void reconnect(), 'Reconnect camera');
      return;
    }
    showCard(null);
    pump?.resetReference();
    pump?.start();
    setPhase('running');
  }

  function pause(reason: 'user' | 'background'): void {
    if (phase !== 'running') return;
    pump?.stop();
    clearOverlay();
    if (reason === 'user') setPhase('paused');
    else setPhase('interrupted');
  }

  function resume(): void {
    if (phase !== 'paused' && phase !== 'interrupted') return;
    if (camera.state !== 'running') {
      void reconnect();
      return;
    }
    showCard(null);
    pump?.start();
    setPhase('running');
  }

  function onVisibility(): void {
    if (closed) return;
    if (document.hidden) {
      if (phase === 'running') pause('background');
      return;
    }
    if (phase === 'interrupted') {
      showCard(
        infoCard({
          icon: 'camera',
          title: 'Capture paused',
          text: 'The camera stops while the app is in the background. Point at the area you captured last, then resume.',
          actions: [button({ label: 'Resume capture', icon: 'play', variant: 'primary', size: 'lg', block: true, onClick: resume, testId: 'resume' })],
        }),
      );
    }
  }

  async function toggleTorch(force?: boolean): Promise<void> {
    const want = force ?? !torchOn;
    const ok = await camera.setTorch(want);
    if (!ok) {
      if (force === undefined) toast('The torch is not available on this camera.', { tone: 'warn' });
      return;
    }
    torchOn = want;
    torchBtn.setAttribute('aria-pressed', String(want));
    torchBtn.classList.toggle('active', want);
    torchBtn.replaceChildren(icon(want ? 'torch' : 'torchOff', { size: 22 }));
  }

  function updateGps(): void {
    if (!geo) return;
    const label = gpsChip.querySelector('span:last-child')!;
    const latest = geo.latest();
    gpsChip.classList.toggle('warn', geo.status === 'denied' || geo.approximate);
    if (geo.status === 'denied') label.textContent = 'GPS off';
    else if (!latest) label.textContent = 'GPS…';
    else label.textContent = geo.approximate ? 'Approx.' : formatAccuracy(latest.accuracy);
  }

  function stopSensors(): void {
    pump?.stop();
    camera.stop();
    motion?.stop();
    compass.stop();
    geo?.stop();
  }

  /** Show the Cancel button of the finishing card after this long (a normal finish takes < 1 s). */
  const FINISH_CANCEL_AFTER_MS = 2000;
  /** A cancelled finishCapture that has not stopped after this long is aborted (worker restarted). */
  const FINISH_ABORT_GRACE_MS = 4000;

  async function finish(): Promise<void> {
    if (numKeyframes < MIN_KEYFRAMES || !(phase === 'running' || phase === 'paused')) return;
    setPhase('finishing');
    pump?.stop();
    clearOverlay();
    // Finishing drains the engine's keyframe mapping queue. It can be cancelled: a finishCapture
    // that ignores the cancel (stuck worker) is aborted and the worker restarted with the
    // persisted capture (EngineLike.abortLongOperation).
    let abortP: Promise<AbortOutcome> | null = null;
    const detail = h('p', { class: 'capture-card-text', text: 'Mapping the last keyframes.' });
    const cancelBtn = button({
      label: 'Cancel',
      variant: 'secondary',
      size: 'lg',
      block: true,
      testId: 'finish-cancel',
      onClick: () => {
        if (abortP) return;
        cancelBtn.disabled = true;
        setButtonLabel(cancelBtn, 'Stopping…');
        detail.textContent = 'Stopping…';
        abortP = app.engine.abortLongOperation({ graceMs: FINISH_ABORT_GRACE_MS });
      },
    });
    const actions = h('div', { class: 'capture-card-actions', hidden: true }, cancelBtn);
    showCard(h('div', { class: 'capture-card', role: 'status', 'data-testid': 'capture-finishing' }, spinner('lg'), h('h2', { class: 'capture-card-title', text: 'Finishing capture…' }), detail, actions));
    const revealCancel = setTimeout(() => {
      actions.hidden = false;
      if (!abortP) detail.textContent = 'Still mapping the last keyframes — this can take a while after a long scan.';
    }, FINISH_CANCEL_AFTER_MS);
    const unsubProgress = app.engine.onProgress((p) => {
      if (!abortP && p.message && !actions.hidden) detail.textContent = `${capitalize(p.message)}…`;
    });
    // Let the frame in flight complete (back-pressure: at most one).
    for (let i = 0; i < 50 && app.engine.frameInFlight && !abortP; i++) await sleep(100);
    const durationMs = elapsed();
    stopSensors();
    let cancelled = false;
    if (!abortP) {
      try {
        await app.engine.finishCapture();
      } catch (e) {
        if (isCancelledError(e)) cancelled = true;
        else console.warn('[capture] finishCapture failed', e); // refine maps what is left
      }
    } else cancelled = true;
    clearTimeout(revealCancel);
    unsubProgress();
    const how = abortP ? await (abortP as Promise<AbortOutcome>) : null;
    if (closed) return;
    // Finished before the cancel took effect: continue as usual.
    if (cancelled) showFinishStopped(durationMs, how === 'restarted' ? 'restarted' : 'cancelled');
    else await completeFinish(durationMs);
  }

  /** Session record + live preview for Processing, then navigate there. */
  async function completeFinish(durationMs: number): Promise<void> {
    app.results.liveSparse = latestSparse;
    app.results.liveCameras = latestCameras;
    await saveCaptured(durationMs);
    if (closed) return;
    closed = true;
    app.navigate({ type: 'FINISH_CAPTURE', steps: cfg.autoDense ? ['densify'] : [] });
  }

  async function saveCaptured(durationMs: number): Promise<void> {
    await sessionCreated;
    void app.updateSession(sessionId, {
      status: 'captured',
      stats: {
        keyframes: numKeyframes,
        points: last?.numPoints ?? 0,
        durationMs: Math.round(durationMs),
        quality: cfg.quality,
        ...(camRes ? { camera: camRes.label, resolution: `${camRes.width}×${camRes.height}` } : {}),
      },
    });
  }

  /** finishCapture was cancelled (or aborted with a worker restart): process now, or later from Home. */
  function showFinishStopped(durationMs: number, how: AbortOutcome): void {
    const restarted = how === 'restarted';
    // Without a stored session a restarted engine has nothing left to process.
    const lost = restarted && !sessionId;
    const kf = `${numKeyframes} keyframes`;
    showCard(
      infoCard({
        icon: lost ? 'warning' : 'info',
        tone: lost ? 'bad' : 'neutral',
        title: restarted ? 'Engine restarted' : 'Finishing cancelled',
        text: lost
          ? 'The engine did not respond and was restarted. This scan was not saved on the device, so it cannot be processed.'
          : `${restarted ? 'The engine did not respond and was restarted. ' : ''}Your ${kf} are saved. Process them now — the last keyframes are mapped during processing — or later from the home screen.`,
        testId: 'capture-finish-stopped',
        actions: [
          ...(lost ? [] : [button({ label: 'Process now', icon: 'sparkles', variant: 'primary', size: 'lg', block: true, testId: 'finish-process', onClick: () => void completeFinish(durationMs) })]),
          button({
            label: 'Back to home',
            variant: lost ? 'primary' : 'secondary',
            size: 'lg',
            block: true,
            testId: 'finish-home',
            onClick: () => void (async () => {
              if (!lost) await saveCaptured(durationMs);
              await abandon(lost);
            })(),
          }),
        ],
      }),
    );
  }

  async function cancel(): Promise<void> {
    const wasRunning = phase === 'running';
    if (wasRunning) pause('user');
    const ok = await confirmDialog({
      title: 'Discard this scan?',
      message: numKeyframes > 0 ? 'The photos captured so far will be deleted.' : undefined,
      confirmLabel: 'Discard',
      cancelLabel: 'Keep scanning',
      destructive: true,
    });
    if (!ok) {
      if (wasRunning) resume();
      return;
    }
    await abandon(true);
  }

  async function abandon(deleteSession = numKeyframes < MIN_KEYFRAMES): Promise<void> {
    closed = true;
    stopSensors();
    void app.wakeLock.release();
    // Cancel what the engine is still doing for this capture (a finishCapture that ignores the
    // cancel is aborted and the worker restarted, so the next scan does not queue behind it; a new
    // scan's init waits for that abort — EngineClient). A session that is deleted must not be
    // re-opened by the restart (its replayed init with `resume` would re-create it): no replay, and
    // the deletion waits until the engine has let go of the session (only a stuck step takes long:
    // ≤ the grace period).
    const discard = !!sessionId && deleteSession;
    const aborted = app.engine.abortLongOperation({ graceMs: FINISH_ABORT_GRACE_MS, reinit: !discard }).catch(() => 'none' as const);
    if (sessionId) {
      await sessionCreated;
      if (deleteSession) {
        const slow = setTimeout(() => showCard(infoCard({ spinner: true, title: 'Discarding scan…', text: 'Stopping the engine.', testId: 'capture-discarding' })), 300);
        await aborted;
        clearTimeout(slow);
        await app.deleteSession(sessionId);
      } else await app.updateSession(sessionId, { stats: { keyframes: numKeyframes, points: last?.numPoints ?? 0, quality: cfg.quality } });
    }
    app.loadedSessionId = null;
    app.navigate({ type: 'CANCEL' });
  }

  return {
    el,
    enter() {
      app.debug.holdFrames = (hold) => {
        if (hold) {
          pump?.stop();
          video.pause();
        } else {
          void video.play().catch(() => undefined);
          if (phase === 'running') pump?.start();
        }
      };
      d.add(() => {
        app.debug.holdFrames = undefined;
      });
      d.listen(document, 'visibilitychange', onVisibility);
      d.add(app.engine.onLive(onLive));
      d.add(app.engine.onError(onEngineError));
      if (debugOverlay) d.add(app.engine.onLog((l) => debugOverlay.onLog(l)));
      d.interval(() => {
        timeLabel.textContent = formatClock(elapsed());
      }, 500);
      const onResize = (): void => {
        clearOverlay();
        sizeOverlay();
      };
      d.listen(window, 'resize', onResize);
      d.listen(window, 'orientationchange', onResize);
      if (cfg.cloudView && app.env.webgl) setCloudView(true, false);
      void start();
    },
    leave() {
      closed = true;
      setPhase('closed');
      stopSensors();
      if (viewerTimer) clearTimeout(viewerTimer);
      if (overlayClearTimer) clearTimeout(overlayClearTimer);
      stopFollow();
      viewer?.dispose();
      viewer = null;
      d.dispose();
    },
  };
}

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

/** Resolves true when the camera reaches 'running', false on any other final state / timeout. */
function settledCamera(camera: CameraCapture, timeoutMs: number): Promise<boolean> {
  return new Promise((resolve) => {
    let done = false;
    const finish = (v: boolean): void => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      unsub();
      resolve(v);
    };
    const unsub = camera.onStateChange((s) => {
      if (s === 'running') finish(true);
      else if (s !== 'restarting' && s !== 'starting') finish(false);
    });
    const timer = setTimeout(() => finish(camera.state === 'running'), timeoutMs);
    if (camera.state === 'running') finish(true);
  });
}

function capitalize(s: string): string {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

function cameraCentre(p: { R: ArrayLike<number>; t: ArrayLike<number> }): number[] {
  const R = p.R,
    t = p.t;
  return [-(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]), -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]), -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2])];
}

/** Evenly spread subset of at most `max` points. */
export function decimate(positions: Float32Array, colors: Uint8Array, max: number): { positions: Float32Array; colors: Uint8Array } {
  const n = Math.floor(positions.length / 3);
  if (n <= max) return { positions, colors };
  const step = n / max;
  const p = new Float32Array(max * 3);
  const c = new Uint8Array(max * 3);
  for (let i = 0; i < max; i++) {
    const j = Math.floor(i * step);
    p[3 * i] = positions[3 * j];
    p[3 * i + 1] = positions[3 * j + 1];
    p[3 * i + 2] = positions[3 * j + 2];
    c[3 * i] = colors[3 * j];
    c[3 * i + 1] = colors[3 * j + 1];
    c[3 * i + 2] = colors[3 * j + 2];
  }
  return { positions: p, colors: c };
}

/** At most `max` cameras, always keeping the newest. */
export function decimateCameras<T>(cams: T[], max: number): T[] {
  if (cams.length <= max) return cams;
  const out: T[] = [];
  const step = (cams.length - 1) / (max - 1);
  for (let i = 0; i < max - 1; i++) out.push(cams[Math.floor(i * step)]);
  out.push(cams[cams.length - 1]);
  return out;
}

/** Square JPEG thumbnail of the current video frame (cover crop). */
async function videoThumbnail(video: HTMLVideoElement, size = 240): Promise<Blob | null> {
  try {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return null;
    const c = document.createElement('canvas');
    c.width = size;
    c.height = size;
    const ctx = c.getContext('2d');
    if (!ctx) return null;
    const s = Math.min(vw, vh);
    ctx.drawImage(video, (vw - s) / 2, (vh - s) / 2, s, s, 0, 0, size, size);
    return await new Promise<Blob | null>((r) => c.toBlob((b) => r(b), 'image/jpeg', 0.78));
  } catch {
    return null;
  }
}
