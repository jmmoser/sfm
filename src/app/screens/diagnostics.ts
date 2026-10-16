/**
 * Diagnostics: collectDiagnostics() rendered as readable sections (device, features, WebGPU
 * adapter / limits / features, cameras + capabilities, motion rate & sign self-check, storage,
 * memory hints, Lockdown suspicion) with a "Copy" button (JSON for bug reports). A "Test
 * motion sensors" button requests motion permission from the tap and samples for 3 s.
 */
import { MotionSensor } from '../../sensors/motion';
import { collectDiagnostics, type SensorDiagnostics } from '../../sensors/diagnostics';
import type { App, Screen } from '../app';
import { button, chip, group, iconButton, kvRow, section, spinner } from '../components/controls';
import { toast } from '../components/toast';
import { Disposer, h, replaceChildren } from '../dom';
import { formatBytes } from '../format';
import type { Route } from '../router';

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

export function createDiagnosticsScreen(app: App, _route: Extract<Route, { screen: 'diagnostics' }>): Screen {
  const d = new Disposer();
  let diag: SensorDiagnostics | null = null;
  let closed = false;
  const motionTest: MotionTest = { state: 'idle' };
  const body = h('div', { class: 'diag-body' }, h('div', { class: 'list-loading' }, spinner('md', 'Collecting diagnostics')));
  const copyBtn = button({ label: 'Copy', icon: 'copy', variant: 'ghost', size: 'sm', testId: 'diag-copy', onClick: () => void copy() });
  copyBtn.disabled = true;

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
    h('div', { class: 'content' }, body),
  );

  function report(): Record<string, unknown> {
    return {
      app: { mock: app.mock, engine: app.store.get().engine, env: app.env, version: import.meta.env?.MODE ?? '' },
      diagnostics: diag,
      motionTest,
    };
  }

  async function copy(): Promise<void> {
    const text = JSON.stringify(report(), (_k, v) => (v instanceof Float64Array || v instanceof Float32Array ? Array.from(v) : v), 2);
    try {
      await navigator.clipboard.writeText(text);
      toast('Diagnostics copied', { tone: 'good', icon: 'copy' });
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
    toast(ok ? 'Diagnostics copied' : 'Copy failed — your browser blocked the clipboard', { tone: ok ? 'good' : 'warn' });
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
      d.dispose();
    },
  };
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
