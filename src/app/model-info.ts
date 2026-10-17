/**
 * Pure helpers for the Viewer / Processing screens (unit-tested in model-info.test.ts):
 * dense-resume detection, north reference of the model, engine memory statistics.
 */
import type { QualityPreset } from '../workers/protocol';
import type { Results } from './app';

type Stats = Record<string, number | string>;

/**
 * A densify of this scan was interrupted and can continue from its stored depth maps: the engine
 * reports `stats.denseResumablePass` (loadSession, src/engine: `dense/progress` in the depth
 * store) and the app marks `denseInterrupted` when a densify it ran was cancelled / failed (the
 * engine resumes those the same way). Never once a dense cloud exists or without a sparse model.
 */
export function denseResumable(r: Pick<Results, 'sparse' | 'dense' | 'stats'>): boolean {
  if (!r.sparse || r.dense) return false;
  return typeof r.stats.denseResumablePass === 'number' || !!r.stats.denseInterrupted;
}

/**
 * Quality preset of the interrupted densify, which a resume must use (the engine's resume key
 * covers the settings: another preset starts over): the app's marker of a run it interrupted
 * (`denseInterruptedQuality`, newest), else the engine's (`denseResumableQuality`, loadSession, from
 * the stored dense progress). null = unknown (older records).
 */
export function denseResumeQuality(r: Pick<Results, 'stats'>): QualityPreset | null {
  for (const k of ['denseInterruptedQuality', 'denseResumableQuality']) {
    const v = r.stats[k];
    if (v === 'fast' || v === 'balanced' || v === 'max') return v;
  }
  return null;
}

/** Drop the dense-resume markers (a densify finished, or refine invalidated the stored depth maps). */
export function clearDenseResume(stats: Stats): void {
  delete stats.denseResumablePass;
  delete stats.denseResumableQuality;
  delete stats.denseInterrupted;
  delete stats.denseInterruptedQuality;
}

export interface NorthInfo {
  label: string;
  /** Explanation shown under the row (null: nothing to add). */
  note: string | null;
  tone: 'good' | 'warn' | 'neutral';
}

/**
 * Which north the model's +Y axis points to (ARCHITECTURE §2: ENU when north is known). The
 * world alignment reports `stats.north` ('true' | 'magnetic' | 'none'), `yawSource` ('gps' |
 * 'compass' | 'gps+compass' | 'none'), the WMM `declinationDeg` / `declinationSource` and the yaw
 * σ `yawStdDeg`. The compass (`webkitCompassHeading` = CLHeading.magneticHeading,
 * docs/research/ios-sensors.md) measures MAGNETIC north: without a location for the declination
 * the model is aligned to magnetic north (`northAligned` false, `north` 'magnetic'). Sessions
 * from before those keys: `northAligned` + `yawSource` (compass ⇒ magnetic unless a declination
 * is reported).
 */
export function northInfo(northAligned: boolean | undefined, stats: Stats): NorthInfo {
  const src = typeof stats.yawSource === 'string' ? stats.yawSource : null;
  const decl = typeof stats.declinationDeg === 'number' && Number.isFinite(stats.declinationDeg) ? stats.declinationDeg : null;
  const sigma = typeof stats.yawStdDeg === 'number' && Number.isFinite(stats.yawStdDeg) ? ` (±${stats.yawStdDeg < 1 ? stats.yawStdDeg.toFixed(1) : Math.round(stats.yawStdDeg)}°)` : '';
  const declText = decl !== null ? `${decl >= 0 ? '+' : '−'}${Math.abs(decl).toFixed(1)}°` : null;
  const declSrc = stats.declinationSource === 'gps' ? ' at the GPS location' : stats.declinationSource === 'coarse' ? ' at the approximate location' : '';
  const magneticNote = `From the compass, which measures magnetic north${sigma}. True north differs by the local declination (up to ±20°, e.g. ≈ 15° in Seattle) — no location was known to correct it; enable GPS for the scan.`;
  let north = stats.north === 'true' || stats.north === 'magnetic' || stats.north === 'none' ? stats.north : null;
  if (north === null) {
    // older sessions: infer from northAligned + yawSource. A compass yaw without a declination is
    // MAGNETIC — also with northAligned false, which is how the world alignment reports a
    // magnetic-only yaw (it is applied: +Y ≈ magnetic north), so it is not "unknown".
    if (src === 'compass') north = northAligned && (decl !== null || stats.northReference === 'true') ? 'true' : 'magnetic';
    else if (!northAligned) north = 'none';
    else north = 'true';
  }
  if (north === 'none') return { label: 'Unknown', note: null, tone: 'neutral' };
  if (north === 'magnetic') return { label: 'Magnetic north', note: magneticNote, tone: 'warn' };
  switch (src) {
    case 'gps':
      return { label: 'True north', note: `From the GPS track${sigma}.`, tone: 'good' };
    case 'compass':
      return { label: 'True north', note: `From the compass${sigma}, corrected by the magnetic declination${declText ? ` (${declText}${declSrc})` : ''}.`, tone: 'good' };
    case 'gps+compass':
      return { label: 'True north', note: `GPS track and compass combined${sigma}${declText ? `; declination ${declText}${declSrc}` : ''}.`, tone: 'good' };
    default:
      return { label: 'True north', note: sigma ? `Yaw uncertainty${sigma}.` : null, tone: 'good' };
  }
}

/** Engine memory statistics as (label, value) rows: stats.memoryMB, denseMemPeakMB, size / step caps. */
export function memoryStatRows(st: Stats): [string, string][] {
  const num = (k: string): number | null => {
    const v = st[k];
    return typeof v === 'number' && Number.isFinite(v) ? v : null;
  };
  const mb = (v: number): string => (v >= 1000 ? `${(v / 1000).toFixed(2)} GB` : `${Math.round(v)} MB`);
  const rows: [string, string][] = [];
  const mem = num('memoryMB');
  if (mem !== null) rows.push(['Engine memory (last step)', `≈ ${mb(mem)}`]);
  const peak = num('denseMemPeakMB');
  if (peak !== null) rows.push(['Dense peak (planned)', `≈ ${mb(peak)}`]);
  const size = num('denseMaxSize');
  const req = num('denseMaxSizeRequested');
  if (size !== null) rows.push(['Dense image size', req !== null && req > size ? `${size} px (${req} px requested — reduced to fit memory)` : `${size} px`]);
  const step = num('fusionStep');
  if (step !== null && step > 1) rows.push(['Fusion pixel step', `${step} (memory)`]);
  return rows;
}
