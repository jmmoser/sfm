/**
 * Setup: capture mode (object / large area / auto) with tips, quality preset with time and
 * memory notes, toggles (GPS, torch, dense model after capture) and ONE "Start" button whose
 * tap handler synchronously requests motion (iOS: must happen inside the gesture), then the
 * camera — then navigates to Capture.
 */
import type { CaptureMode, QualityPreset } from '../../workers/protocol';
import type { App, Screen } from '../app';
import { button, choiceCard, iconButton, section, segmented, switchRow, group } from '../components/controls';
import { Disposer, h, replaceChildren } from '../dom';
import { canScan } from '../env';
import { icon, type IconName } from '../icons';

interface ModeInfo {
  mode: CaptureMode;
  icon: IconName;
  title: string;
  subtitle: string;
  tips: string[];
}

const MODES: ModeInfo[] = [
  {
    mode: 'object',
    icon: 'object',
    title: 'Object',
    subtitle: 'Things from shoe to statue size — walk around them.',
    tips: [
      'Move around the object at a constant distance; do not just turn the phone.',
      'Keep 70 % or more overlap between views — move slowly and steadily.',
      'Do a second, higher loop looking down to capture the top.',
      'Avoid shiny or transparent surfaces; matte, textured objects work best.',
    ],
  },
  {
    mode: 'scene',
    icon: 'area',
    title: 'Large area',
    subtitle: 'Rooms, gardens, facades — walk through the space.',
    tips: [
      'Walk slowly and sidestep; avoid turning on the spot.',
      'Look at every structure from several places, not from just one point.',
      'End where you started to close the loop — it removes drift.',
      'Use GPS outdoors for real-world scale and north alignment.',
    ],
  },
  {
    mode: 'auto',
    icon: 'auto',
    title: 'Auto',
    subtitle: 'Picks object or area settings from what it sees.',
    tips: ['Start by moving the phone slowly sideways.', 'Follow the on-screen guidance while capturing.'],
  },
];

const QUALITY_NOTES: Record<QualityPreset, string> = {
  fast: 'Quickest. About 1–3 min of processing for 50 photos, ~400 MB memory. Good for previews.',
  balanced: 'Recommended. About 3–8 min of processing for 50 photos, ~700 MB memory.',
  max: 'Most detail. 10+ min of processing, ~1.1 GB memory — use a recent iPhone and keep the app open.',
};

export function createSetupScreen(app: App): Screen {
  const d = new Disposer();
  let cfg = { ...app.config };
  const modeList = h('div', { class: 'choice-list', role: 'radiogroup', 'aria-label': 'What are you scanning?' });
  const tipsHost = h('div', { class: 'tips-host' });
  const qualityNote = h('p', { class: 'quality-note', 'aria-live': 'polite' });

  const gpsRow = switchRow({
    label: 'Use location (GPS)',
    description: 'Real-world scale and north for outdoor scans',
    icon: 'location',
    checked: cfg.useGps,
    testId: 'toggle-gps',
    onChange: (v) => {
      cfg.useGps = v;
      app.saveConfig({ useGps: v });
    },
  });
  const torchRow = switchRow({
    label: 'Torch',
    description: 'Light for close objects in dim places (if available)',
    icon: 'torch',
    checked: cfg.torch,
    testId: 'toggle-torch',
    onChange: (v) => {
      cfg.torch = v;
      app.config.torch = v;
    },
  });
  const denseRow = switchRow({
    label: 'Dense model after capture',
    description: 'Build the detailed point cloud right away',
    icon: 'points',
    checked: cfg.autoDense,
    testId: 'toggle-dense',
    onChange: (v) => {
      cfg.autoDense = v;
      app.saveConfig({ autoDense: v });
    },
  });

  function renderModes(): void {
    replaceChildren(
      modeList,
      ...MODES.map((m) =>
        choiceCard({
          icon: m.icon,
          title: m.title,
          subtitle: m.subtitle,
          selected: cfg.mode === m.mode,
          testId: `mode-${m.mode}`,
          onSelect: () => selectMode(m.mode),
        }),
      ),
    );
    const info = MODES.find((m) => m.mode === cfg.mode) ?? MODES[0];
    replaceChildren(
      tipsHost,
      h(
        'div',
        { class: 'tips' },
        h('div', { class: 'tips-head' }, icon('info', { size: 18 }), h('span', { text: `Tips for ${info.title.toLowerCase()} scans` })),
        h('ul', { class: 'tips-list' }, ...info.tips.map((t) => h('li', { text: t }))),
      ),
    );
  }

  function selectMode(mode: CaptureMode): void {
    if (mode === cfg.mode) return;
    cfg.mode = mode;
    // GPS defaults on for large areas, off for objects (indoors, no use at 1 m scale).
    cfg.useGps = mode === 'scene';
    gpsRow.input.checked = cfg.useGps;
    app.saveConfig({ mode, useGps: cfg.useGps });
    renderModes();
  }

  const quality = segmented<QualityPreset>(
    'Quality',
    [
      { value: 'fast', label: 'Fast' },
      { value: 'balanced', label: 'Balanced' },
      { value: 'max', label: 'Max' },
    ],
    cfg.quality,
    (q) => {
      cfg.quality = q;
      app.saveConfig({ quality: q });
      qualityNote.textContent = QUALITY_NOTES[q];
    },
  );
  quality.el.dataset.testid = 'quality';
  qualityNote.textContent = QUALITY_NOTES[cfg.quality];

  const start = button({
    label: 'Start capture',
    icon: 'camera',
    variant: 'primary',
    size: 'xl',
    block: true,
    testId: 'start-capture',
    // NO await before requestCapturePermissions: iOS needs the motion request inside the tap.
    onClick: () => {
      if (!canScan(app.env)) return;
      app.config = { ...app.config, ...cfg };
      app.requestCapturePermissions();
      app.startNewScan();
    },
  });
  start.disabled = !canScan(app.env);

  const el = h(
    'div',
    { class: 'screen-scroll setup' },
    h(
      'header',
      { class: 'topbar' },
      iconButton('back', 'Back', () => app.navigate({ type: 'BACK' }), { testId: 'back' }),
      h('h1', { class: 'topbar-title', text: 'New scan' }),
      h('span', { class: 'topbar-spacer' }),
    ),
    h(
      'div',
      { class: 'content content-cta' },
      section('What are you scanning?', modeList, tipsHost),
      section('Quality', quality.el, qualityNote),
      section('Options', group(gpsRow.el, torchRow.el, denseRow.el)),
    ),
    h(
      'div',
      { class: 'cta-bar' },
      start,
      h('p', { class: 'cta-note' }, icon('lock', { size: 14 }), h('span', { text: 'Camera & motion access · photos stay on this device' })),
    ),
  );

  return {
    el,
    enter() {
      renderModes();
      d.add(app.store.select((s) => s.env, (env) => (start.disabled = !canScan(env))));
    },
    leave() {
      d.dispose();
    },
  };
}
