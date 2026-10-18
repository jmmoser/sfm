/**
 * Diagnostics: collectDiagnostics() rendered as readable sections (device, features, WebGPU
 * adapter / limits / features, cameras + capabilities, motion rate & sign self-check, storage,
 * memory hints, Lockdown suspicion) with a "Copy" button (JSON for bug reports). A "Test
 * motion sensors" button requests motion permission from the tap and samples for 3 s.
 *
 * "Benchmark & self-test" (top): runs src/app/bench in a dedicated worker — WebGPU SIFT,
 * matcher, PatchMatch and bundle adjustment timings with GPU-vs-CPU parity / ground-truth
 * checks (pass / fail chips), scan-time estimates per quality preset, Cancel, "Copy report"
 * (text + JSON). CPU-only results without WebGPU. `?benchScale=small` = CI-sized inputs. The run
 * is cancelled when the page is hidden (iOS drops GPU work in the background) or the screen left.
 *
 * "Camera timing": opens the camera from the tap and runs src/sensors/camera-probe.ts — a quick
 * 10 s run (rotate the phone once: orientation and rotation transients) or a 60 s still run
 * (drift) — rVFC captureTime vs MediaStreamTrackProcessor timestamps, buffer orientation, and the
 * main-thread cost of createImageBitmap(<video>) (full / downscaled) and new VideoFrame(<video>).
 * The report is part of "Copy".
 */
import { startBenchmark, type BenchRun } from '../bench/bench';
import { STATUS_LABEL, estimateLine, formatMs, qualityName, reportClipboardText, verdict } from '../bench/report';
import { describeTest } from '../bench/scales';
import { BENCH_TESTS, type BenchReport, type BenchScale, type BenchStatus, type BenchTestId, type BenchTestResult } from '../bench/types';
import { CameraCapture } from '../../sensors/camera';
import { runCameraTimingProbe, type CameraProbeReport } from '../../sensors/camera-probe';
import { MotionSensor } from '../../sensors/motion';
import { collectDiagnostics, diagnosticsReportJSON, type SensorDiagnostics } from '../../sensors/diagnostics';
import type { App, Screen } from '../app';
import { banner, button, chip, group, iconButton, kvRow, section, spinner } from '../components/controls';
import { actionSheet } from '../components/sheet';
import { toast } from '../components/toast';
import { Disposer, h, replaceChildren } from '../dom';
import { icon } from '../icons';
import { formatBytes, formatDuration, type Tone } from '../format';
import type { Route } from '../router';

type ProbeMode = 'quick' | 'still';

const PROBE_MODES: Record<ProbeMode, { durationMs: number; warmupMs: number; label: string; hint: string }> = {
  quick: { durationMs: 15_000, warmupMs: 2_000, label: 'Quick probe (15 s, rotate)', hint: 'Point at something textured, then turn the phone to landscape, to the other landscape side, and back to portrait.' },
  still: { durationMs: 60_000, warmupMs: 5_000, label: 'Still probe (60 s, for drift)', hint: 'Hold the phone still (lean it against something) until it finishes.' },
};

interface CameraProbeState {
  state: 'idle' | 'running' | 'done' | 'error';
  mode?: ProbeMode;
  phase?: 'warmup' | 'measure';
  remainingMs?: number;
  report?: CameraProbeReport;
  error?: string;
}

interface MotionTest {
  state: 'idle' | 'running' | 'done' | 'denied' | 'unsupported';
  rateHz?: number;
  samples?: number;
  calibration?: Record<string, unknown>;
}

const KEY_LIMITS = [
  'maxBufferSize',
  'maxStorageBufferBindingSize',
  'maxComputeWorkgroupStorageSize',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupSizeX',
  'maxStorageBuffersPerShaderStage',
  'maxTextureDimension2D',
  'maxTextureArrayLayers',
  'maxBindGroups',
];
const BYTE_LIMITS = new Set(['maxBufferSize', 'maxStorageBufferBindingSize', 'maxComputeWorkgroupStorageSize']);

/** The last benchmark report (kept while the app runs: leaving Diagnostics does not lose it). */
let lastBenchReport: BenchReport | null = null;

/** `?benchScale=small` (CI) — otherwise the full, device-sized inputs. */
function benchScaleFromUrl(): BenchScale {
  try {
    const v = new URLSearchParams(location.search).get('benchScale');
    return v === 'small' || v === 'tiny' ? v : 'full';
  } catch {
    return 'full';
  }
}

const STATUS_TONE: Record<BenchStatus, Tone> = { pending: 'neutral', running: 'info', pass: 'good', fail: 'bad', error: 'bad', skipped: 'neutral', cancelled: 'warn' };

export function createDiagnosticsScreen(app: App, _route: Extract<Route, { screen: 'diagnostics' }>): Screen {
  const d = new Disposer();
  let diag: SensorDiagnostics | null = null;
  let closed = false;
  const motionTest: MotionTest = { state: 'idle' };
  const probe: CameraProbeState = { state: 'idle' };
  const probeVideo = h('video', { class: 'probe-video', playsinline: true, muted: true, autoplay: true, 'aria-label': 'Camera preview' });
  let probeCamera: CameraCapture | null = null;
  const body = h('div', { class: 'diag-body' }, h('div', { class: 'list-loading' }, spinner('md', 'Collecting diagnostics')));
  const copyBtn = button({ label: 'Copy', icon: 'copy', variant: 'ghost', size: 'sm', testId: 'diag-copy', onClick: () => void copy() });
  copyBtn.disabled = true;
  const benchHost = h('div', { class: 'bench', 'data-testid': 'bench' });

  const el = h(
    'div',
    { class: 'screen-scroll diagnostics' },
    h(
      'header',
      { class: 'topbar' },
      iconButton('back', 'Back', () => app.navigate({ type: 'BACK' }), { testId: 'back' }),
      h('h1', { class: 'topbar-title', text: 'Diagnostics' }),
      copyBtn,
    ),
    h('div', { class: 'content' }, benchHost, body),
  );

  function report(): Record<string, unknown> {
    return {
      app: { mock: app.mock, engine: app.store.get().engine, env: app.env, version: import.meta.env?.MODE ?? '' },
      diagnostics: diag,
      motionTest,
      cameraProbe: probe.report ?? (probe.error ? { error: probe.error } : null),
    };
  }

  async function copy(): Promise<void> {
    // camera device ids and any location are redacted (diagnosticsReportJSON)
    await copyText(diagnosticsReportJSON(report()), 'Diagnostics copied');
  }

  // ---- benchmark ---------------------------------------------------------------------------------
  const benchScale = benchScaleFromUrl();
  let benchRun: BenchRun | null = null;
  const benchTests = new Map<BenchTestId, BenchTestResult>();
  const benchProgress = new Map<BenchTestId, string>();
  let benchStarted = 0;
  let benchTimer: ReturnType<typeof setInterval> | null = null;

  function runBench(): void {
    if (benchRun) return;
    benchTests.clear();
    benchProgress.clear();
    lastBenchReport = null;
    app.debug.benchReport = undefined;
    benchStarted = performance.now();
    // GPU only when this page has WebGPU (the worker checks too and falls back to the CPU).
    const gpu = !!(navigator as Navigator & { gpu?: unknown }).gpu;
    const run = startBenchmark({
      scale: benchScale,
      gpu,
      noGpuReason: gpu ? undefined : 'WebGPU is not available in this browser (older iOS, Lockdown Mode or a disabled feature)',
      onTest: (r) => {
        benchTests.set(r.id, r);
        if (r.status !== 'running') benchProgress.delete(r.id);
        renderBench();
      },
      onProgress: (id, m) => {
        benchProgress.set(id, m);
        renderBench();
      },
    });
    benchRun = run;
    benchTimer = setInterval(() => renderBenchClock(), 1000);
    void run.done.then((rep) => {
      if (benchRun !== run) return;
      benchRun = null;
      if (benchTimer) clearInterval(benchTimer);
      benchTimer = null;
      lastBenchReport = rep;
      app.debug.benchReport = rep;
      // the final report is authoritative (a terminated worker never sent its last updates)
      for (const t of rep.tests) benchTests.set(t.id, t);
      benchProgress.clear();
      if (!closed) {
        renderBench();
        const v = verdict(rep.tests);
        if (rep.cancelled) toast('Benchmark cancelled', { tone: 'warn' });
        else toast(v.fail ? `Benchmark finished: ${v.text}` : 'Benchmark finished — all checks passed', { tone: v.fail ? 'bad' : 'good' });
      }
    });
    renderBench();
  }

  /**
   * Run the benchmark — after asking when the processing engine still holds a scan: the benchmark
   * allocates hundreds of MB (GPU buffers, a second WebGPU device) and, together with a loaded scan,
   * can exceed the iPhone tab's memory limit (iOS then closes the app without warning).
   */
  function requestBench(): void {
    if (benchRun) return;
    if (!app.engineHoldsData()) {
      runBench();
      return;
    }
    actionSheet(
      'A scan is loaded in memory',
      [
        {
          label: 'Free memory and run',
          icon: 'activity',
          description: 'Unloads the scan from the processing engine first. It stays saved on this device.',
          testId: 'bench-release-run',
          onSelect: () => void releaseAndRun(),
        },
        {
          label: 'Run anyway',
          icon: 'warning',
          description: 'iOS may close the app if memory runs out.',
          testId: 'bench-run-anyway',
          onSelect: () => runBench(),
        },
      ],
      'The benchmark needs a lot of memory. With a scan loaded as well, iOS may close the app.',
    );
  }

  async function releaseAndRun(): Promise<void> {
    try {
      await app.releaseEngine();
    } catch (e) {
      toast(`Could not unload the scan: ${(e as Error)?.message ?? e}`, { tone: 'bad' });
      return;
    }
    if (!closed) runBench();
  }

  function cancelBench(reason?: string): void {
    benchRun?.cancel(reason);
    renderBench();
  }

  let benchClock: HTMLElement | null = null;
  function renderBenchClock(): void {
    if (benchClock && benchRun) benchClock.textContent = `Running · ${formatDuration(performance.now() - benchStarted)}`;
  }

  function benchRow(id: BenchTestId, title: string): HTMLElement {
    const t = benchTests.get(id);
    const idle = !benchRun && !t;
    const status: BenchStatus = t?.status ?? 'pending';
    const progress = benchProgress.get(id);
    const sub =
      status === 'running'
        ? (progress ?? 'Starting…')
        : t?.error
          ? `Error: ${t.error}`
          : t?.summary || t?.note || (idle || status === 'pending' ? describeTest(id, benchScale) : '');
    const badge = idle
      ? null
      : status === 'running'
        ? h('span', { class: 'bench-chip-running' }, spinner('sm', `${title} running`))
        : chip(STATUS_LABEL[status], STATUS_TONE[status], status === 'pass' ? 'check' : status === 'fail' || status === 'error' ? 'warning' : undefined);
    const details: HTMLElement[] = [];
    if (t && (status === 'pass' || status === 'fail' || status === 'error' || status === 'cancelled')) {
      // (the "no WebGPU — CPU fallback" note of every test is said once, in the verdict banner)
      if (t.note && t.summary && t.backend === 'gpu') details.push(h('div', { class: 'bench-note' }, icon('info', { size: 14 }), h('span', { text: t.note })));
      if (t.checks.length)
        details.push(
          h(
            'ul',
            { class: 'bench-checks', 'aria-label': `${title} checks` },
            ...t.checks.map((c) => h('li', { class: c.pass ? 'ok' : 'bad' }, icon(c.pass ? 'check' : 'warning', { size: 14, label: c.pass ? 'passed' : 'failed' }), h('span', { text: `${c.name}: ${c.detail}` }))),
          ),
        );
      const tm = Object.entries(t.timings);
      if (tm.length)
        details.push(
          h(
            'details',
            { class: 'bench-timings' },
            h('summary', { text: `Timings (${tm.length})` }),
            h('dl', null, ...tm.flatMap(([k, v]) => [h('dt', { text: k }), h('dd', { text: formatMs(v) })])),
          ),
        );
    }
    return h(
      'div',
      { class: 'row bench-row', 'data-testid': `bench-${id}`, 'data-status': idle ? 'idle' : status },
      h(
        'div',
        { class: 'bench-head' },
        h(
          'div',
          { class: 'row-text' },
          h('span', { class: 'row-title' }, h('span', { text: title }), t ? h('span', { class: 'bench-backend', text: t.backend.toUpperCase() }) : null),
          sub ? h('span', { class: 'row-sub', text: sub }) : null,
        ),
        badge,
      ),
      details.length ? h('div', { class: 'bench-body' }, ...details) : null,
    );
  }

  function renderBench(): void {
    if (closed) return;
    const rep = benchRun ? null : lastBenchReport;
    const parts: HTMLElement[] = [];
    parts.push(
      h(
        'p',
        { class: 'muted bench-intro' },
        `Measures this device on synthetic data — GPU features, matching, dense stereo and bundle adjustment — and checks the GPU results against the CPU reference. ${
          benchScale === 'full' ? 'Takes about 1–3 minutes; keep the app open.' : `Scale “${benchScale}” (CI-sized inputs).`
        }`,
      ),
    );
    if (rep) {
      const v = verdict(rep.tests);
      const g = rep.device.gpu;
      parts.push(
        banner({
          tone: rep.error && !rep.cancelled ? 'bad' : rep.cancelled ? 'warn' : v.fail ? 'bad' : 'good',
          icon: rep.cancelled || v.fail ? 'warning' : 'check',
          title: rep.cancelled ? `Cancelled — ${v.text}` : v.fail ? `${v.fail} of ${rep.tests.length} tests failed` : `All ${v.pass} tests passed`,
          body: `${g.available ? `WebGPU: ${g.adapter}${g.isSoftware ? ' (software renderer)' : ''}` : `No WebGPU — CPU results only (${g.reason ?? 'unavailable'})`} · ${formatDuration(rep.durationMs)}${rep.error ? ` · ${rep.error}` : ''}`,
          testId: 'bench-verdict',
        }),
      );
    }
    parts.push(group(...BENCH_TESTS.map((t) => benchRow(t.id, t.title))));
    if (rep?.estimates.length) {
      const e0 = rep.estimates[0];
      parts.push(
        h('h3', { class: 'bench-subtitle', text: 'Estimated processing time' }),
        group(
          ...rep.estimates.map((e) =>
            h(
              'div',
              { class: 'row bench-row', 'data-testid': `bench-estimate-${e.quality}` },
              h('div', { class: 'row-text' }, h('span', { class: 'row-title', text: qualityName(e.quality) }), h('span', { class: 'row-sub', text: estimateLine(e) })),
              chip(`≈ ${formatDuration(e.processingMs)}`, 'info'),
            ),
          ),
        ),
        h(
          'p',
          { class: 'muted bench-foot' },
          `Rough estimate for a scan of ${e0.keyframes} keyframes, after the capture (refine + dense + mesh); features on the ${e0.features.toUpperCase()}, dense stereo on the ${e0.dense.toUpperCase()}. Extrapolated from the measurements above.`,
        ),
      );
    }
    const actions: HTMLElement[] = [];
    if (benchRun) {
      benchClock = h('span', { class: 'bench-clock', 'aria-live': 'off' });
      actions.push(h('div', { class: 'bench-running' }, spinner('sm', 'Benchmark running'), benchClock));
      actions.push(button({ label: 'Cancel', icon: 'close', variant: 'secondary', size: 'md', block: true, testId: 'bench-cancel', onClick: () => cancelBench() }));
      renderBenchClock();
    } else {
      benchClock = null;
      actions.push(button({ label: rep ? 'Run again' : 'Run benchmark', icon: 'activity', variant: 'primary', size: 'md', block: true, testId: 'bench-run', onClick: () => requestBench() }));
      if (rep)
        actions.push(
          button({ label: 'Copy report', icon: 'copy', variant: 'secondary', size: 'md', block: true, testId: 'bench-copy', onClick: () => void copyText(reportClipboardText(rep), 'Benchmark report copied') }),
        );
    }
    parts.push(h('div', { class: 'bench-actions' }, ...actions));
    const focused = benchHost.contains(document.activeElement) ? (document.activeElement as HTMLElement).dataset.testid : undefined;
    // keep the open/closed state of the timing lists across re-renders
    const open = new Set([...benchHost.querySelectorAll<HTMLDetailsElement>('details[open]')].map((d) => d.closest<HTMLElement>('[data-testid]')?.dataset.testid));
    replaceChildren(benchHost, section('Benchmark & self-test', ...parts));
    for (const id of open) if (id) benchHost.querySelector<HTMLDetailsElement>(`[data-testid="${id}"] details`)?.setAttribute('open', '');
    if (focused) {
      const next = benchHost.querySelector<HTMLElement>(`[data-testid="${focused}"]`) ?? benchHost.querySelector<HTMLElement>(benchRun ? '[data-testid="bench-cancel"]' : '[data-testid="bench-run"]');
      next?.focus({ preventScroll: true });
    }
  }

  function yesNo(v: boolean | null | undefined): HTMLElement {
    if (v === null || v === undefined) return chip('Unknown', 'neutral');
    return v ? chip('Yes', 'good') : chip('No', 'neutral');
  }

  function testMotion(): void {
    // Synchronous inside the tap (iOS requires a user gesture for the permission request).
    const perm = MotionSensor.requestPermission();
    motionTest.state = 'running';
    render();
    void perm.then(async (p) => {
      if (p !== 'granted') {
        motionTest.state = p === 'unsupported' ? 'unsupported' : 'denied';
        render();
        return;
      }
      const m = new MotionSensor();
      m.start();
      await new Promise((r) => setTimeout(r, 3000));
      motionTest.rateHz = m.rateHz;
      motionTest.samples = m.sampleCount;
      const cal = m.calibration;
      motionTest.calibration = {
        accelSign: cal.accelSign,
        accelSignSource: cal.accelSignSource,
        linearAccelSignSource: cal.linearAccelSignSource,
        gyroSource: cal.gyroSource,
        version: cal.version,
      };
      m.stop();
      motionTest.state = 'done';
      if (!closed) render();
    });
  }

  function runProbe(mode: ProbeMode): void {
    if (probe.state === 'running') return;
    const cfg = PROBE_MODES[mode];
    probe.state = 'running';
    probe.mode = mode;
    probe.phase = 'warmup';
    probe.remainingMs = cfg.warmupMs;
    probe.report = undefined;
    probe.error = undefined;
    render();
    const cam = new CameraCapture();
    probeCamera = cam;
    // start() inside the tap: iOS asks for (or reuses) the camera grant from the gesture
    void cam
      .start(probeVideo, { quality: 'balanced' })
      .then(() =>
        runCameraTimingProbe(cam, {
          durationMs: cfg.durationMs,
          warmupMs: cfg.warmupMs,
          isCancelled: () => closed,
          onProgress: (ms, phase) => {
            if (phase !== probe.phase || Math.ceil(ms / 1000) !== Math.ceil((probe.remainingMs ?? 0) / 1000)) {
              probe.phase = phase;
              probe.remainingMs = ms;
              if (!closed) render();
            }
          },
        }),
      )
      .then(
        (r) => {
          probe.report = r;
          probe.state = 'done';
        },
        (e) => {
          probe.error = (e as Error)?.message ?? String(e);
          probe.state = 'error';
        },
      )
      .finally(() => {
        cam.stop();
        if (probeCamera === cam) probeCamera = null;
        if (!closed) render();
      });
  }

  function probeSection(): HTMLElement {
    const rows: HTMLElement[] = [];
    const r = probe.report;
    const running = probe.state === 'running';
    if (running && probe.mode) rows.push(h('p', { class: 'muted', text: PROBE_MODES[probe.mode].hint }), probeVideo);
    if (probe.state === 'error') rows.push(kvRow('Probe', chip(probe.error ?? 'Failed', 'bad')));
    if (r) {
      const t = r.timing;
      const rot = r.rotation;
      const mono = (text: string): HTMLElement => h('span', { class: 'mono', text });
      const grabRow = (title: string, g: CameraProbeReport['grab']['bitmap']): HTMLElement =>
        kvRow(title, mono(g ? `p50 ${formatMs(g.syncMs.p50)} · p95 ${formatMs(g.syncMs.p95)}` : '–'));
      rows.push(
        group(
          kvRow('Measured', mono(`${formatMs(r.durationMs)} after ${formatMs(r.warmupMs)} warm-up`)),
          kvRow('rVFC frames', mono(`${r.video.frames} (${r.video.fps} fps, ${r.video.timeSource})`)),
          kvRow('MSTP frames', r.mstp.available ? mono(`${r.mstp.frames} (${r.mstp.fps} fps, ${r.mstp.transport ?? '?'})`) : chip('Unavailable', 'warn')),
          kvRow('Timestamp jitter', mono(t.jitterStdMs === null ? '–' : `σ ${t.jitterStdMs} ms · p5–p95 ${t.jitterRangeMs} ms`)),
          kvRow('Timestamp drift', mono(t.driftMsPerS === null ? '–' : `${t.driftMsPerS} ms/s (${t.pairs} frames, ${t.segments} segment${t.segments === 1 ? '' : 's'})`)),
          kvRow('Rotations', mono(rot.changes ? `${rot.changes} · dropped ${rot.videoDropped} / MSTP ${rot.mstpDropped} · misoriented ${rot.videoMisoriented}` : 'none')),
          kvRow('MSTP buffers', mono(r.mstp.codedSizes.join(', ') || '–')),
          grabRow('Bitmap, full', r.grab.bitmap),
          grabRow('Bitmap, downscaled', r.grab.bitmapScaled),
          grabRow('VideoFrame', r.grab.videoFrame),
        ),
        ...r.summary.map((s) => h('p', { class: 'muted', text: s })),
      );
    }
    const buttons = (['quick', 'still'] as const).map((mode) =>
      button({
        label: running && probe.mode === mode ? `${probe.phase === 'warmup' ? 'Warming up' : 'Measuring'}… ${Math.ceil((probe.remainingMs ?? 0) / 1000)} s` : PROBE_MODES[mode].label,
        icon: 'camera',
        variant: 'secondary',
        size: 'md',
        block: true,
        disabled: running,
        testId: `diag-camera-probe-${mode}`,
        onClick: () => runProbe(mode),
      }),
    );
    return section('Camera timing', ...rows, ...buttons);
  }

  function render(): void {
    if (!diag) return;
    const g = diag;
    const sections: HTMLElement[] = [];

    if (g.lockdownSuspected) {
      sections.push(
        h('div', { class: 'banner banner-bad', role: 'alert' }, h('div', { class: 'banner-text' }, h('div', { class: 'banner-title', text: 'Lockdown Mode suspected' }), h('div', { class: 'banner-body', text: 'WebGPU, camera and IndexedDB are all unavailable.' }))),
      );
    }

    sections.push(
      section(
        'Device',
        group(
          kvRow('Platform', h('span', { text: g.platform || '–' })),
          kvRow('Apple WebKit', yesNo(g.appleMobileWebKit)),
          kvRow('Home Screen app', yesNo(g.standalone)),
          kvRow('Secure context', yesNo(g.secureContext)),
          kvRow('Cross-origin isolated', yesNo(g.crossOriginIsolated)),
          kvRow('CPU cores (reported)', h('span', { text: g.hardwareConcurrency === null ? '–' : String(g.hardwareConcurrency) })),
          kvRow('Screen', h('span', { text: `${g.screen.width}×${g.screen.height} @${g.devicePixelRatio}x · θ ${g.screen.screenAngle}°` })),
          h('div', { class: 'row row-block' }, h('span', { class: 'row-title', text: 'User agent' }), h('code', { class: 'code-inline', text: g.userAgent })),
        ),
      ),
    );

    sections.push(
      section(
        'Features',
        h(
          'div',
          { class: 'feature-grid' },
          ...Object.entries(g.features)
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([k, v]) => h('span', { class: `feature ${v ? 'on' : 'off'}`, 'aria-label': `${k}: ${v ? 'yes' : 'no'}` }, h('span', { class: 'feature-dot' }), h('span', { text: k }))),
        ),
      ),
    );

    const gpu = g.gpu;
    const gpuRows: HTMLElement[] = [kvRow('WebGPU', gpu?.available ? chip('Available', 'good') : chip(g.features.webgpu ? 'No adapter' : 'Not supported', 'warn'))];
    if (gpu?.available) {
      const info = gpu.info;
      if (info) gpuRows.push(kvRow('Adapter', h('span', { text: [info.vendor, info.architecture, info.description].filter(Boolean).join(' · ') || '–' })));
      if (gpu.isSoftware !== undefined) gpuRows.push(kvRow('Software renderer', yesNo(gpu.isSoftware)));
      for (const k of KEY_LIMITS) {
        const v = gpu.limits?.[k];
        if (typeof v === 'number') gpuRows.push(kvRow(camelBreaks(k), h('span', { class: 'mono', text: BYTE_LIMITS.has(k) ? `${formatBytes(v)} (${v})` : String(v) })));
      }
      if (gpu.preferredCanvasFormat) gpuRows.push(kvRow('Canvas format', h('span', { class: 'mono', text: gpu.preferredCanvasFormat })));
    }
    const gpuSection = section('WebGPU', group(...gpuRows));
    if (gpu?.features?.length) gpuSection.appendChild(h('div', { class: 'tag-list' }, ...gpu.features.map((f) => h('span', { class: 'tag', text: f }))));
    if (gpu?.wgslLanguageFeatures?.length) gpuSection.appendChild(h('div', { class: 'tag-list' }, ...gpu.wgslLanguageFeatures.map((f) => h('span', { class: 'tag tag-dim', text: f }))));
    sections.push(gpuSection);

    const cams = g.cameras ?? [];
    sections.push(
      section(
        'Cameras',
        cams.length
          ? group(
              ...cams.map((c) => {
                const caps = c.capabilities ?? {};
                const w = caps.width as { max?: number } | undefined;
                const hh = caps.height as { max?: number } | undefined;
                const zoom = caps.zoom as { min?: number; max?: number } | undefined;
                const fd = caps.focusDistance as { min?: number } | undefined;
                const bits = [
                  w?.max && hh?.max ? `max ${w.max}×${hh.max}` : null,
                  zoom ? `zoom ${zoom.min ?? '?'}–${zoom.max ?? '?'}` : null,
                  fd?.min !== undefined ? `focus ≥ ${fd.min} m` : null,
                  caps.torch ? 'torch' : null,
                ].filter(Boolean);
                return h(
                  'div',
                  { class: 'row row-camera' },
                  h('span', { class: 'row-text' }, h('span', { class: 'row-title', text: c.label || '(label hidden until camera access is granted)' }), h('span', { class: 'row-sub', text: bits.join(' · ') || '–' })),
                  chip(c.lens, c.lens === 'wide' ? 'good' : c.lens === 'virtual' ? 'warn' : 'neutral'),
                );
              }),
            )
          : h('p', { class: 'muted', text: 'No cameras listed (permission not granted yet, or no camera).' }),
      ),
    );

    const mt = motionTest;
    const motionRows: HTMLElement[] = [
      kvRow('DeviceMotion', yesNo(g.features.deviceMotion)),
      kvRow('Permission API (iOS)', yesNo(g.features.motionPermissionApi)),
    ];
    if (mt.state === 'done') {
      motionRows.push(kvRow('Event rate', h('span', { text: `${(mt.rateHz ?? 0).toFixed(1)} Hz (${mt.samples ?? 0} samples)` })));
      const cal = mt.calibration ?? {};
      motionRows.push(kvRow('Accel sign', h('span', { class: 'mono', text: `${cal.accelSign} (${cal.accelSignSource})` })));
      motionRows.push(kvRow('Gyro axes', h('span', { class: 'mono', text: String(cal.gyroSource) })));
      if (!mt.rateHz) motionRows.push(h('p', { class: 'muted', text: 'No motion events — expected on desktop browsers.' }));
    } else if (mt.state === 'denied') motionRows.push(kvRow('Test', chip('Permission denied', 'warn')));
    else if (mt.state === 'unsupported') motionRows.push(kvRow('Test', chip('Unsupported', 'warn')));
    const motionBtn = button({
      label: mt.state === 'running' ? 'Sampling for 3 s…' : 'Test motion sensors',
      icon: 'motion',
      variant: 'secondary',
      size: 'md',
      block: true,
      disabled: mt.state === 'running',
      testId: 'diag-motion',
      onClick: testMotion,
    });
    sections.push(section('Motion', group(...motionRows), motionBtn));
    sections.push(probeSection());

    const s = g.storage;
    sections.push(
      section(
        'Storage',
        group(
          kvRow('Used', h('span', { text: s ? formatBytes(s.usage) : '–' })),
          kvRow('Quota', h('span', { text: s ? formatBytes(s.quota) : '–' })),
          kvRow('Persistent', yesNo(s?.persisted ?? null)),
          kvRow('IndexedDB', yesNo(g.features.indexedDB)),
          kvRow('OPFS', yesNo(g.features.opfs)),
        ),
      ),
    );

    sections.push(section('Memory', group(...memoryHints(g))));
    if (g.errors.length) sections.push(section('Probe errors', h('pre', { class: 'code-block', text: g.errors.join('\n') })));
    // Re-rendering replaces the focused control (e.g. the motion test button): keep focus on it.
    const focused = body.contains(document.activeElement) ? (document.activeElement as HTMLElement).dataset.testid : undefined;
    replaceChildren(body, ...sections);
    if (focused) body.querySelector<HTMLElement>(`[data-testid="${focused}"]`)?.focus({ preventScroll: true });
  }

  return {
    el,
    enter() {
      // Coming back to Diagnostics: show the previous run's results.
      if (lastBenchReport) for (const t of lastBenchReport.tests) benchTests.set(t.id, t);
      renderBench();
      // iOS drops GPU work submitted while the page is hidden: the timings would be meaningless.
      d.listen(document, 'visibilitychange', () => {
        if (document.hidden && benchRun) cancelBench('the app went to the background');
      });
      void collectDiagnostics({ gpu: 'adapter' })
        .then((r) => {
          if (closed) return;
          diag = r;
          copyBtn.disabled = false;
          render();
        })
        .catch((e) => {
          if (!closed) replaceChildren(body, h('p', { class: 'muted', text: `Diagnostics failed: ${(e as Error)?.message ?? e}` }));
        });
    },
    leave() {
      closed = true;
      benchRun?.cancel('Diagnostics was closed');
      probeCamera?.stop();
      if (benchTimer) clearInterval(benchTimer);
      d.dispose();
    },
  };
}

/** Copy text to the clipboard (Clipboard API, else a selected textarea + execCommand) with a toast. */
async function copyText(text: string, okMessage: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text);
    toast(okMessage, { tone: 'good', icon: 'copy' });
    return;
  } catch {
    /* fall back */
  }
  const ta = h('textarea', { class: 'offscreen', readonly: true, 'aria-hidden': 'true' });
  ta.value = text;
  document.body.appendChild(ta);
  ta.select();
  let ok = false;
  try {
    ok = document.execCommand('copy');
  } catch {
    ok = false;
  }
  ta.remove();
  toast(ok ? okMessage : 'Copy failed — your browser blocked the clipboard', { tone: ok ? 'good' : 'warn' });
}

/** Insert zero-width break opportunities at camelCase boundaries (long WebGPU limit names). */
function camelBreaks(s: string): string {
  return s.replace(/([a-z0-9])([A-Z])/g, '$1\u200b$2');
}

/** RAM is not exposed on iOS; WebGPU maxBufferSize is a coarse proxy (ios-platform.md §1.1). */
export function memoryHints(g: Pick<SensorDiagnostics, 'deviceMemoryGB' | 'gpu' | 'appleMobileWebKit'>): HTMLElement[] {
  const rows: HTMLElement[] = [kvRow('Device memory API', h('span', { text: g.deviceMemoryGB !== null ? `${g.deviceMemoryGB} GB` : 'Not exposed' }))];
  const est = estimateRamGB(g.gpu?.limits?.maxBufferSize, g.appleMobileWebKit);
  if (est !== null) rows.push(kvRow('Estimated RAM (from GPU limits)', h('span', { text: `≈ ${est} GB` })));
  const budget = est === null ? null : est <= 4 ? 0.6 : est <= 6 ? 0.9 : 1.2;
  if (g.appleMobileWebKit) rows.push(kvRow('Memory budget', h('span', { text: budget ? `≈ ${budget} GB for the whole tab` : '≈ 1.2 GB (iPhone limit)' })));
  return rows;
}

/** maxBufferSize ≈ min(1 GiB, 0.08 · RAM) on iOS 26.2+; null when not applicable. */
export function estimateRamGB(maxBufferSize: number | undefined, apple: boolean): number | null {
  if (!apple || !maxBufferSize || maxBufferSize >= 1024 ** 3 || maxBufferSize <= 256 * 1024 ** 2) return null;
  const gb = maxBufferSize / 0.08 / 1024 ** 3;
  const classes = [4, 6, 8, 12];
  return classes.reduce((best, c) => (Math.abs(c - gb) < Math.abs(best - gb) ? c : best), classes[0]);
}
