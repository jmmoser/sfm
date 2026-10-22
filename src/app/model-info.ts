/**
 * Pure helpers for the Viewer / Processing screens (unit-tested in model-info.test.ts):
 * dense / splat resume detection, the photoreal build card (progress line, preview caption,
 * "Use it now"), north reference of the model, engine memory statistics.
 */
import { SPLAT_STAGES, type QualityPreset } from '../workers/protocol';
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

/**
 * Quality preset the dense cloud / mesh / splats were built with (`denseQuality` / `meshQuality`
 * set by the Processing screen, `splatQuality` by the viewer's photoreal build). Older records
 * lack them: the scan's preset is the best guess.
 */
export function builtQuality(r: Pick<Results, 'stats' | 'quality'>, what: 'dense' | 'mesh' | 'splat'): QualityPreset {
  const v = r.stats[what === 'dense' ? 'denseQuality' : what === 'mesh' ? 'meshQuality' : 'splatQuality'];
  return v === 'fast' || v === 'balanced' || v === 'max' ? v : r.quality;
}

/**
 * An interrupted splat training can continue from its checkpoint: the engine reports
 * `stats.splatResumableIteration` (loadSession) and the app marks `splatInterrupted` when a build
 * it ran was cancelled / failed. Needs a sparse model; an existing splat model does not prevent
 * it (an interrupted rebuild at another quality).
 */
export function splatResumable(r: Pick<Results, 'sparse' | 'stats'>): boolean {
  if (!r.sparse) return false;
  return typeof r.stats.splatResumableIteration === 'number' || !!r.stats.splatInterrupted;
}

/** Quality of the interrupted splat run (app marker first, else the engine's), null when unknown. */
export function splatResumeQuality(r: Pick<Results, 'stats'>): QualityPreset | null {
  for (const k of ['splatInterruptedQuality', 'splatResumableQuality']) {
    const v = r.stats[k];
    if (v === 'fast' || v === 'balanced' || v === 'max') return v;
  }
  return null;
}

/**
 * The stored splat model was ended early ("Use it now") and its training can continue from the
 * checkpoint the engine kept at the model's iteration: `stats.splatContinuesModel` (set by the
 * viewer with splatResumableIteration from the result's splatContinueFrom; by loadSession after a
 * reload only when the stored checkpoint is that run's, not another interrupted run's at the same
 * iteration), `splatResumableIteration` equals `splat.iterations`, and no build of the app was
 * interrupted since. Not an interruption: no "interrupted" card; the Build sheet offers to
 * continue the training.
 */
export function splatContinuable(r: Pick<Results, 'sparse' | 'splat' | 'stats'>): boolean {
  if (!r.sparse || !r.splat || r.stats.splatInterrupted || r.stats.splatContinuesModel !== 1) return false;
  return r.stats.splatResumableIteration === r.splat.iterations;
}

/**
 * Stats after a photoreal build of the app at quality `q` was cancelled / failed: interrupted,
 * resumable at `q`. Which checkpoint the engine holds now is not known here (a cancel ≥ 100
 * iterations after the last one writes another), so the iteration reported earlier (loadSession,
 * or "Use it now" before a continued run) and the continue marker go: the card names no iteration.
 */
export function splatInterruptedStats(stats: Stats, q: QualityPreset): Stats {
  const out: Stats = { ...stats, splatInterrupted: 1, splatInterruptedQuality: q };
  delete out.splatResumableIteration;
  delete out.splatResumableQuality;
  delete out.splatContinuesModel;
  return out;
}

/** "1,250 of 3,000 iterations" (`of` part only when the schedule is known and longer). */
export function splatIterationsText(iteration: number, iterations?: unknown): string {
  const it = iteration.toLocaleString('en-US');
  return typeof iterations === 'number' && iterations > iteration ? `${it} of ${iterations.toLocaleString('en-US')} iterations` : `${it} iterations`;
}

/**
 * The Build sheet's photoreal item at quality `q` when it is more than Build / Rebuild / "Built":
 * Resume (an interrupted run at `q`), Continue (the stored model was ended early and its checkpoint
 * kept) or — not "Built" — Rebuild of a model ended early at `q` whose checkpoint is gone (a new
 * dense cloud, another run's checkpoint): it is incomplete and starts over. null: the plain item.
 */
export function splatBuildResume(r: Pick<Results, 'sparse' | 'splat' | 'stats' | 'quality'>, q: QualityPreset): { label: string; sub: string } | null {
  const rq = splatResumeQuality(r);
  if (splatResumable(r) && (!rq || rq === q)) {
    if (splatContinuable(r) && r.splat) return { label: 'Continue', sub: `Stopped early at ${splatIterationsText(r.splat.iterations, r.splat.stats.splatIterations)} — continues the training.` };
    return { label: 'Resume', sub: 'Interrupted — continues from the saved training.' };
  }
  if (r.sparse && r.splat?.stats.splatFinishedEarly === 1 && builtQuality(r, 'splat') === q)
    return { label: 'Rebuild', sub: `Stopped early at ${splatIterationsText(r.splat.iterations, r.splat.stats.splatIterations)} — retrains from the start.` };
  return null;
}

/**
 * Drop the splat-resume markers (a training finished, or refine / densify invalidated the
 * checkpoints: a new pose set or dense cloud is a new run).
 */
export function clearSplatResume(stats: Stats): void {
  delete stats.splatResumableIteration;
  delete stats.splatResumableQuality;
  delete stats.splatContinuesModel;
  delete stats.splatInterrupted;
  delete stats.splatInterruptedQuality;
}

export interface SplatProgressView {
  /** Overall 0..1 over the three stages (prepare ≈ 8 %, training ≈ 89 %, finishing ≈ 3 %), −1 = indeterminate. */
  progress: number;
  /** One line for the progress card. */
  text: string;
}

/**
 * The viewer's photoreal-build card from an engine progress event of `splat` (stages
 * SPLAT_STAGES; training events carry a SplatTrainingDetail): overall progress and a line short
 * enough for a phone, such as "1,250 / 3,000 iterations · ~2 min left" (from the it/s).
 */
export function splatProgressView(e: { stage: string; progress: number; message?: string; detail?: Record<string, number> }): SplatProgressView {
  const p = Number.isFinite(e.progress) && e.progress >= 0 ? Math.min(1, e.progress) : -1;
  if (e.stage === SPLAT_STAGES.training) {
    const d = e.detail;
    if (d && typeof d.iteration === 'number' && typeof d.iterations === 'number' && d.iterations > 0) {
      const it = Math.min(d.iteration, d.iterations);
      const parts = [`${it.toLocaleString('en-US')} / ${d.iterations.toLocaleString('en-US')} iterations`];
      const rate = typeof d.itPerSec === 'number' && d.itPerSec > 0 ? d.itPerSec : 0;
      if (rate && it > 0) {
        const s = (d.iterations - it) / rate;
        parts.push(s < 45 ? '< 1 min left' : `~${Math.round(s / 60)} min left`);
      }
      return { progress: 0.08 + 0.89 * (it / d.iterations), text: parts.join(' · ') };
    }
    return { progress: p < 0 ? -1 : 0.08 + 0.89 * p, text: 'Training' };
  }
  if (e.stage === SPLAT_STAGES.finishing) return { progress: 0.97 + 0.03 * Math.max(0, p), text: 'Saving the model' };
  return { progress: p < 0 ? -1 : 0.08 * p, text: 'Preparing the photos' };
}

/** Caption of a 'splat_preview' snapshot (its SplatPreviewDetail): "Iteration 1,250 of 3,000". */
export function splatPreviewCaption(detail?: Record<string, number>): string {
  const it = detail?.iteration;
  if (typeof it !== 'number') return 'Preview';
  if (it <= 0) return 'Initial Gaussians, before training';
  const n = detail?.iterations;
  return `Iteration ${it.toLocaleString('en-US')}${typeof n === 'number' && n > it ? ` of ${n.toLocaleString('en-US')}` : ''}`;
}

export interface SplatUseNow {
  /** "Use it now" can end the training (the engine's finishFrom is reached). */
  ready: boolean;
  /** Why not yet (empty when ready or unknown). */
  note: string;
}

/**
 * "Use it now" on the photoreal build card from the latest 'splat_training' detail: enabled from
 * `finishFrom` (the end of the trainer's coarse-to-fine schedule) on; before the first report
 * (preparation) disabled without a note.
 */
export function splatUseNow(detail?: Record<string, number>): SplatUseNow {
  const it = detail?.iteration;
  const from = detail?.finishFrom;
  if (typeof it !== 'number' || typeof from !== 'number') return { ready: false, note: '' };
  if (it >= from) return { ready: true, note: '' };
  return { ready: false, note: `Usable from iteration ${from.toLocaleString('en-US')}` };
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
