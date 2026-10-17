/**
 * Capture debug overlay: what the engine sees per frame (backend, frame size, feature input,
 * features vs the keyframe minimum, sharpness, luma — a blank frame shows as ≈ 0 ± 0 — gyro,
 * inliers, decision reason, timings), the camera stream, and the latest engine warnings.
 * Enabled with the "Debug overlay" option of the setup screen. "Copy" puts the panel text on
 * the clipboard for bug reports; tapping the panel collapses it.
 */
import type { LiveUpdate, LogMessage } from '../workers/protocol';
import { h } from './dom';

export interface DebugCameraInfo {
  width: number;
  height: number;
  frameRate: number;
  label: string;
  /** grabFrame({maxSide}) of the pump (null = native size). */
  frameMaxSide: number | null;
  /** Frame path ('video', 'mstp', or 'video (MSTP fallback: …)'). */
  path?: string;
}

export interface DebugSnapshot {
  update: LiveUpdate | null;
  camera: DebugCameraInfo | null;
  /** Frames processed per second (engine round trips). */
  fps: number;
  /** navigator.gpu exists on the page. */
  webgpuApi: boolean;
  logs: readonly LogMessage[];
}

const MAX_LOGS = 6;

function num(v: number | null | undefined, digits = 0): string {
  return typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : '–';
}

/** One line per item; a leading "!" marks a line that points at a problem. */
export function formatDebugLines(s: DebugSnapshot): string[] {
  const lines: string[] = [];
  const u = s.update;
  const d = u?.debug;
  const cam = s.camera;
  lines.push(
    `backend: ${d ? (d.backend === 'webgpu' ? 'WebGPU' : 'CPU') : '–'}${d?.fallbackReason ? ` (${d.fallbackReason})` : ''}${s.webgpuApi ? '' : ' · no navigator.gpu'}`,
  );
  if (cam) lines.push(`camera: ${cam.width}×${cam.height} @ ${num(cam.frameRate)} fps${cam.frameMaxSide ? ` → ≤${cam.frameMaxSide}px` : ''} · ${cam.label || 'unnamed'}${cam.path ? ` · ${cam.path}` : ''}`);
  if (!u) {
    lines.push('waiting for the first frame…');
  } else {
    lines.push(`frame #${u.frameIndex}: ${d && d.srcWidth ? `${d.srcWidth}×${d.srcHeight}${d.width && (d.width !== d.srcWidth || d.height !== d.srcHeight) ? ` → ${d.width}×${d.height}` : ''}` : '–'} · ${num(s.fps, 1)} fps processed`);
    if (d) {
      const low = d.numFeatures !== null && d.numFeatures < d.minFeatures;
      lines.push(`${low ? '!' : ''}features: ${d.numFeatures ?? 'not extracted'} / min ${d.minFeatures}${d.featureInput ? ` · input ${d.featureInput}` : ''}`);
      const blank = Number.isFinite(d.lumaStd) && d.lumaStd < 2;
      lines.push(`${blank ? '!' : ''}luma: ${num(d.lumaMean)} ± ${num(d.lumaStd, 1)}${blank ? ' (blank frame?)' : ''}`);
      lines.push(`sharpness: ${num(d.sharpness, 1)} (ref ${num(d.sharpnessRef, 1)}) · gyro ${num(d.angularSpeed, 2)} rad/s`);
      lines.push(`inliers to last KF: ${num(d.inliers)}`);
    }
    lines.push(`decision: ${u.decision} · guidance ${u.guidance} · ${u.tracking}`);
    if (d?.reason) lines.push(`reason: ${d.reason}`);
    const t = u.timings ?? {};
    lines.push(`ms: total ${num(t.total)} · prep ${num(t.prep)} · features ${num(t.features)}`);
  }
  for (const l of s.logs) lines.push(`${l.level === 'warn' || l.level === 'error' ? '!' : ''}[${l.level}] ${l.message}`);
  return lines;
}

export interface DebugOverlay {
  el: HTMLElement;
  setCamera(c: DebugCameraInfo | null): void;
  onLive(u: LiveUpdate): void;
  onLog(l: LogMessage): void;
  text(): string;
}

export function createDebugOverlay(): DebugOverlay {
  const body = h('div', { class: 'debug-overlay-body' });
  const copyBtn = h('button', { type: 'button', class: 'debug-overlay-copy', text: 'Copy' });
  const el = h('div', { class: 'debug-overlay', 'data-testid': 'debug-overlay' }, copyBtn, body);
  let update: LiveUpdate | null = null;
  let camera: DebugCameraInfo | null = null;
  const logs: LogMessage[] = [];
  const times: number[] = [];
  let pending = false;

  const snapshot = (): DebugSnapshot => {
    const fps = times.length >= 2 ? ((times.length - 1) * 1000) / (times[times.length - 1] - times[0]) : NaN;
    return { update, camera, fps, webgpuApi: typeof navigator !== 'undefined' && 'gpu' in navigator, logs };
  };
  const render = (): void => {
    pending = false;
    body.replaceChildren(...formatDebugLines(snapshot()).map((l) => (l.startsWith('!') ? h('div', { class: 'debug-overlay-bad', text: l.slice(1) }) : h('div', { text: l }))));
  };
  const schedule = (): void => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(render);
  };
  const text = (): string => formatDebugLines(snapshot()).map((l) => (l.startsWith('!') ? `! ${l.slice(1)}` : l)).join('\n');

  el.addEventListener('click', () => el.classList.toggle('collapsed'));
  copyBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    void navigator.clipboard?.writeText(`${text()}\n${navigator.userAgent}`).then(
      () => (copyBtn.textContent = 'Copied'),
      () => (copyBtn.textContent = 'Copy failed'),
    );
    setTimeout(() => (copyBtn.textContent = 'Copy'), 1500);
  });
  render();
  return {
    el,
    setCamera(c) {
      camera = c;
      schedule();
    },
    onLive(u) {
      update = u;
      times.push(performance.now());
      if (times.length > 20) times.shift();
      schedule();
    },
    onLog(l) {
      if (l.level === 'debug') return;
      logs.push(l);
      if (logs.length > MAX_LOGS) logs.shift();
      schedule();
    },
    text,
  };
}
