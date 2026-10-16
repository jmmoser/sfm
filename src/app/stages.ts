/**
 * Processing stages shown on the Processing screen, the mapping of engine progress stage
 * names onto them, and a simple ETA model.
 */
import type { ProcessingStep } from './router';
import type { QualityPreset } from '../workers/protocol';

export type StageId = 'matching' | 'reconstruction' | 'alignment' | 'depth_maps' | 'fusion' | 'meshing';

export interface StageDef {
  id: StageId;
  label: string;
  detail: string;
  step: ProcessingStep;
}

export const STAGES: readonly StageDef[] = [
  { id: 'matching', label: 'Matching', detail: 'Finding overlaps between photos', step: 'refine' },
  { id: 'reconstruction', label: 'Reconstruction', detail: 'Solving camera poses and 3D points', step: 'refine' },
  { id: 'alignment', label: 'Alignment', detail: 'Gravity, scale and north from sensors', step: 'refine' },
  { id: 'depth_maps', label: 'Depth maps', detail: 'Multi-view stereo per photo', step: 'densify' },
  { id: 'fusion', label: 'Fusion', detail: 'Merging depth maps into a dense cloud', step: 'densify' },
  { id: 'meshing', label: 'Meshing', detail: 'Building a textured surface', step: 'mesh' },
];

const RULES: [RegExp, StageId][] = [
  [/mesh|tsdf|marching|simplif|surface/i, 'meshing'],
  [/fus/i, 'fusion'],
  [/depth|patch\s*match|patchmatch|mvs|stereo|undistort|view.?select|dense/i, 'depth_maps'],
  [/align|gravity|world|gps|geo|inertial|scale|north|compass/i, 'alignment'],
  [/match|pair|retriev|vocab|verif|feature|extract/i, 'matching'],
  [/recon|sfm|mapper|regist|triang|bundle|\bba\b|refine|incremental|retri?angulat|filter/i, 'reconstruction'],
];

/** Map an engine stage name (e.g. 'matching', 'ba', 'patchmatch', 'tsdf') to a UI stage. */
export function stageIdFor(stage: string): StageId | null {
  const s = stage.trim();
  if (!s) return null;
  const exact = STAGES.find((d) => d.id === s || d.label.toLowerCase() === s.toLowerCase());
  if (exact) return exact.id;
  for (const [re, id] of RULES) if (re.test(s)) return id;
  return null;
}

export function stagesForSteps(steps: readonly ProcessingStep[]): StageId[] {
  return STAGES.filter((d) => steps.includes(d.step)).map((d) => d.id);
}

/** Relative stage durations per quality preset (rough priors; refined by measured progress). */
export function stageWeights(quality: QualityPreset): Record<StageId, number> {
  const dense = quality === 'fast' ? 1 : quality === 'balanced' ? 1.8 : 3;
  return {
    matching: 1,
    reconstruction: 1.4,
    alignment: 0.2,
    depth_maps: 4 * dense,
    fusion: 1.2 * dense,
    meshing: 1.5 * dense,
  };
}

export type StageStatus = 'pending' | 'active' | 'done' | 'skipped' | 'error';

export interface StageState {
  id: StageId;
  status: StageStatus;
  /** 0..1, or -1 for indeterminate. */
  progress: number;
  startedAt: number | null;
  endedAt: number | null;
  message?: string;
}

/**
 * Remaining time (ms) of the scheduled stages, or null while there is not enough information.
 * Model: duration ∝ weight; the time per weight unit is measured from finished stages and the
 * progress of the active stage (prefer measured stages; the active one alone needs ≥ 5 %
 * progress and 1.5 s).
 */
export function estimateRemainingMs(stages: readonly StageState[], weights: Record<StageId, number>, now: number): number | null {
  let doneWeight = 0;
  let doneTime = 0;
  let active: StageState | null = null;
  for (const s of stages) {
    if (s.status === 'done' && s.startedAt !== null && s.endedAt !== null) {
      doneWeight += weights[s.id];
      doneTime += s.endedAt - s.startedAt;
    } else if (s.status === 'active') active = s;
  }
  let activeElapsed = 0;
  let activeFrac = 0;
  if (active && active.startedAt !== null) {
    activeElapsed = Math.max(0, now - active.startedAt);
    activeFrac = active.progress > 0 ? Math.min(1, active.progress) : 0;
  }
  let perWeight: number | null = null;
  if (doneWeight > 0 && doneTime > 0) {
    perWeight = doneTime / doneWeight;
    // Blend in the active stage once it has progressed (its own rate is the freshest signal).
    if (active && activeFrac >= 0.05 && activeElapsed > 1500) {
      const w = weights[active.id];
      const rate = activeElapsed / (activeFrac * w);
      perWeight = (perWeight * doneWeight + rate * w * activeFrac) / (doneWeight + w * activeFrac);
    }
  } else if (active && activeFrac >= 0.05 && activeElapsed > 1500) {
    perWeight = activeElapsed / (activeFrac * weights[active.id]);
  }
  if (perWeight === null || !Number.isFinite(perWeight)) return null;
  let remaining = 0;
  for (const s of stages) {
    if (s.status === 'pending') remaining += weights[s.id] * perWeight;
    else if (s.status === 'active') {
      const total = weights[s.id] * perWeight;
      if (activeFrac > 0 && activeElapsed > 0) remaining += Math.max(0, (activeElapsed / activeFrac) * (1 - activeFrac));
      else remaining += Math.max(0, total - activeElapsed);
    }
  }
  return remaining;
}
