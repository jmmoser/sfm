/**
 * Pure formatting helpers for the UI (no DOM). Numbers use decimal (SI) byte units like iOS.
 */
import type { SessionStatus } from '../io/storage';
import type { CaptureMode, Guidance, QualityPreset, TrackingState } from '../workers/protocol';

const BYTE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB'];

/** 0 B, 950 B, 1.2 KB, 34 MB, 1.25 GB (SI, like iOS Settings). */
export function formatBytes(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n) || n < 0) return '–';
  if (n < 1000) return `${Math.round(n)} B`;
  let v = n;
  let u = 0;
  while (v >= 1000 && u < BYTE_UNITS.length - 1) {
    v /= 1000;
    u++;
  }
  const digits = v >= 100 ? 0 : v >= 10 ? 1 : 2;
  // Round first so 999.96 KB becomes "1.00 MB" instead of "1000 KB".
  let r = Number(v.toFixed(digits));
  if (r >= 1000 && u < BYTE_UNITS.length - 1) {
    r = r / 1000;
    u++;
    return `${r.toFixed(2)} ${BYTE_UNITS[u]}`;
  }
  return `${r.toFixed(digits)} ${BYTE_UNITS[u]}`;
}

/** Clock-style duration: 0:05, 1:23, 12:03, 1:02:03. */
export function formatClock(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) ms = 0;
  const total = Math.floor(ms / 1000);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const ss = String(s).padStart(2, '0');
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`;
}

/** Human duration: 0.4 s, 12 s, 3 min 5 s, 1 h 4 min. */
export function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '–';
  if (ms < 1000) return `${(ms / 1000).toFixed(1)} s`;
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  if (m < 60) {
    const rs = s % 60;
    return rs ? `${m} min ${rs} s` : `${m} min`;
  }
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return rm ? `${h} h ${rm} min` : `${h} h`;
}

/** Remaining-time phrase for an ETA (null = unknown). */
export function formatEta(ms: number | null): string {
  if (ms === null || !Number.isFinite(ms)) return 'Estimating time…';
  if (ms < 5000) return 'Almost done';
  if (ms < 60_000) return 'Less than a minute left';
  const m = Math.round(ms / 60_000);
  if (m < 60) return `About ${m} min left`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return rm ? `About ${h} h ${rm} min left` : `About ${h} h left`;
}

/** 950, 3.4k, 12k, 1.25M. */
export function formatCount(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return '–';
  const a = Math.abs(n);
  if (a < 1000) return String(Math.round(n));
  if (a < 1e6) {
    const k = n / 1000;
    return `${Math.abs(k) < 10 ? trimZeros(k.toFixed(1)) : Math.round(k)}k`;
  }
  const m = n / 1e6;
  return `${Math.abs(m) < 10 ? trimZeros(m.toFixed(2)) : Math.abs(m) < 100 ? trimZeros(m.toFixed(1)) : Math.round(m)}M`;
}

function trimZeros(s: string): string {
  return s.includes('.') ? s.replace(/\.?0+$/, '') : s;
}

/** Plural helper: plural(1, 'frame') = '1 frame', plural(3, 'frame') = '3 frames'. */
export function plural(n: number, word: string, pluralWord = `${word}s`): string {
  return `${n.toLocaleString('en-US')} ${n === 1 ? word : pluralWord}`;
}

/** GPS accuracy: ±4 m, ±35 m, ±1.2 km. */
export function formatAccuracy(m: number | null | undefined): string {
  if (m === null || m === undefined || !Number.isFinite(m)) return '–';
  if (m < 10) return `±${m.toFixed(m < 1 ? 1 : 0)} m`;
  if (m < 1000) return `±${Math.round(m)} m`;
  return `±${(m / 1000).toFixed(1)} km`;
}

export function formatPercent(p: number, digits = 0): string {
  if (!Number.isFinite(p)) return '–';
  return `${(Math.max(0, Math.min(1, p)) * 100).toFixed(digits)}%`;
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

function hhmm(d: Date): string {
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

function startOfDay(d: Date): number {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
}

/** "Just now", "5 min ago", "Today, 14:05", "Yesterday, 09:12", "Oct 12", "Oct 12, 2025". Local time. */
export function formatRelativeDate(epochMs: number, nowMs: number = Date.now()): string {
  if (!Number.isFinite(epochMs)) return '–';
  const d = new Date(epochMs);
  const now = new Date(nowMs);
  const diff = nowMs - epochMs;
  if (diff >= -60_000 && diff < 60_000) return 'Just now';
  if (diff > 0 && diff < 3_600_000) return `${Math.floor(diff / 60_000)} min ago`;
  const days = Math.round((startOfDay(now) - startOfDay(d)) / 86_400_000);
  if (days === 0) return `Today, ${hhmm(d)}`;
  if (days === 1) return `Yesterday, ${hhmm(d)}`;
  const md = `${MONTHS[d.getMonth()]} ${d.getDate()}`;
  return d.getFullYear() === now.getFullYear() ? md : `${md}, ${d.getFullYear()}`;
}

/** Default session name, e.g. "Object · Oct 16, 14:05". */
export function defaultSessionName(mode: CaptureMode, epochMs: number = Date.now()): string {
  const d = new Date(epochMs);
  const label = mode === 'object' ? 'Object' : mode === 'scene' ? 'Area' : 'Scan';
  return `${label} · ${MONTHS[d.getMonth()]} ${d.getDate()}, ${hhmm(d)}`;
}

export type Tone = 'neutral' | 'accent' | 'good' | 'warn' | 'bad' | 'info';

export function statusInfo(status: SessionStatus): { label: string; tone: Tone } {
  switch (status) {
    case 'capturing':
      return { label: 'Incomplete', tone: 'warn' };
    case 'captured':
      return { label: 'Captured', tone: 'info' };
    case 'sparse':
      return { label: 'Sparse', tone: 'neutral' };
    case 'dense':
      return { label: 'Dense', tone: 'accent' };
    case 'meshed':
      return { label: 'Mesh', tone: 'good' };
    default:
      return { label: String(status), tone: 'neutral' };
  }
}

export function modeLabel(mode: CaptureMode): string {
  return mode === 'object' ? 'Object' : mode === 'scene' ? 'Large area' : 'Auto';
}

export function qualityLabel(q: QualityPreset): string {
  return q === 'fast' ? 'Fast' : q === 'balanced' ? 'Balanced' : 'Max';
}

export interface GuidanceInfo {
  title: string;
  detail: string;
  tone: Tone;
}

/** Friendly copy for each guidance code (object vs large-area wording where it matters). */
export function guidanceInfo(g: Guidance, mode: CaptureMode = 'auto'): GuidanceInfo {
  switch (g) {
    case 'good':
      return { title: 'Looking good', detail: 'Keep moving slowly and steadily', tone: 'good' };
    case 'move_slower':
      return { title: 'Slow down', detail: 'Move more slowly to avoid blurry frames', tone: 'warn' };
    case 'more_texture':
      return { title: 'Not enough detail', detail: 'Aim at textured surfaces, avoid plain walls and sky', tone: 'warn' };
    case 'translate_more':
      return mode === 'object'
        ? { title: 'Walk around it', detail: 'Move the phone around the object instead of turning it', tone: 'warn' }
        : { title: 'Move sideways', detail: 'Take a step to the side instead of only turning the phone', tone: 'warn' };
    case 'tracking_lost':
      return { title: 'Tracking lost', detail: 'Go back to a part you already captured', tone: 'bad' };
    case 'initializing':
      return mode === 'object'
        ? { title: 'Getting started', detail: 'Slowly move around the object to begin', tone: 'info' }
        : { title: 'Getting started', detail: 'Move the phone slowly sideways to begin', tone: 'info' };
    default:
      return { title: '', detail: '', tone: 'neutral' };
  }
}

export function trackingLabel(t: TrackingState): string {
  return t === 'tracking' ? 'Tracking' : t === 'lost' ? 'Lost' : 'Starting';
}

/** Escape text for safe insertion into HTML strings. */
export function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c] as string);
}

/**
 * File name for an export, e.g. "Object_Oct_16_14_05.ply". Letters and digits of every script are
 * kept ("花瓶.ply", "Кухня.glb"; Latin accents are dropped: "Küche" → "Kuche"); punctuation, emoji
 * and path characters become separators.
 */
export function exportFilename(sessionName: string, ext: string, suffix = ''): string {
  const base =
    Array.from(
      sessionName
        .normalize('NFKD')
        .replace(/(\p{Script=Latin})\p{M}+/gu, '$1') // Café → Cafe (portable); が, ё, 한 keep their marks
        .normalize('NFC')
        .replace(/[^\p{L}\p{M}\p{N}\s_-]+/gu, ' ')
        .trim()
        .replace(/\s+/g, '_')
        .replace(/_+/g, '_'),
    )
      .slice(0, 60)
      .join('')
      .replace(/^[_-]+|[_-]+$/g, '') || 'scan';
  return `${base}${suffix ? `_${suffix}` : ''}.${ext.replace(/^\./, '')}`;
}
