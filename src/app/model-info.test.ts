import { describe, expect, it } from 'vitest';
import type { SparseResult, SplatResult } from '../workers/protocol';
import {
  builtQuality,
  clearDenseResume,
  clearSplatResume,
  denseResumable,
  denseResumeQuality,
  memoryStatRows,
  northInfo,
  splatContinuable,
  splatIterationsText,
  splatPreviewCaption,
  splatProgressView,
  splatResumable,
  splatResumeQuality,
  splatUseNow,
} from './model-info';

const sparse = { points: { count: 0, positions: new Float32Array(0), colors: new Uint8Array(0) }, cameras: [], stats: {} } as unknown as SparseResult;
const cloud = { count: 1, positions: new Float32Array(3), colors: new Uint8Array(3) };

describe('denseResumable', () => {
  it('engine-reported resumable pass (loadSession) or an interrupted densify of this app', () => {
    expect(denseResumable({ sparse, dense: null, stats: { denseResumablePass: 1 } })).toBe(true);
    expect(denseResumable({ sparse, dense: null, stats: { denseResumablePass: 0 } })).toBe(true); // pass 0 is a pass
    expect(denseResumable({ sparse, dense: null, stats: { denseInterrupted: 1 } })).toBe(true);
    expect(denseResumable({ sparse, dense: null, stats: {} })).toBe(false);
  });
  it('never with a dense cloud or without a sparse model', () => {
    expect(denseResumable({ sparse, dense: cloud, stats: { denseResumablePass: 1 } })).toBe(false);
    expect(denseResumable({ sparse: null, dense: null, stats: { denseResumablePass: 1 } })).toBe(false);
  });
});

describe('denseResumeQuality', () => {
  it('the interrupted run’s preset: the app marker first (newest), else the engine’s; null when unknown', () => {
    expect(denseResumeQuality({ stats: { denseResumablePass: 1, denseResumableQuality: 'max' } })).toBe('max');
    expect(denseResumeQuality({ stats: { denseInterrupted: 1, denseInterruptedQuality: 'fast', denseResumableQuality: 'max' } })).toBe('fast');
    expect(denseResumeQuality({ stats: { denseResumablePass: 1 } })).toBeNull();
    expect(denseResumeQuality({ stats: { denseResumableQuality: 'ultra' } })).toBeNull();
  });
  it('clearDenseResume drops every marker', () => {
    const st: Record<string, number | string> = { denseResumablePass: 1, denseResumableQuality: 'max', denseInterrupted: 1, denseInterruptedQuality: 'fast', keyframes: 9 };
    clearDenseResume(st);
    expect(st).toEqual({ keyframes: 9 });
  });
});

describe('builtQuality', () => {
  it('reads the recorded preset per model, else the scan preset', () => {
    const r = { quality: 'balanced' as const, stats: { denseQuality: 'max', meshQuality: 'fast' } };
    expect(builtQuality(r, 'dense')).toBe('max');
    expect(builtQuality(r, 'mesh')).toBe('fast');
    expect(builtQuality({ quality: 'balanced', stats: {} }, 'mesh')).toBe('balanced');
    expect(builtQuality({ quality: 'fast', stats: { denseQuality: 'bogus' } }, 'dense')).toBe('fast');
  });
});

describe('northInfo', () => {
  it('world-alignment stats: north reference, yaw source, declination, σ', () => {
    expect(northInfo(true, { north: 'true', yawSource: 'gps', yawStdDeg: 1.84 })).toEqual({ label: 'True north', note: 'From the GPS track (±2°).', tone: 'good' });
    const c = northInfo(true, { north: 'true', yawSource: 'compass', declinationDeg: -3.25, declinationSource: 'gps', yawStdDeg: 0.6 });
    expect(c.label).toBe('True north');
    expect(c.note).toBe('From the compass (±0.6°), corrected by the magnetic declination (−3.3° at the GPS location).');
    expect(northInfo(true, { north: 'true', yawSource: 'gps+compass', declinationDeg: 15.2, declinationSource: 'coarse' }).note).toBe(
      'GPS track and compass combined; declination +15.2° at the approximate location.',
    );
    // Compass without a location: aligned to MAGNETIC north (northAligned false in the metric info).
    const m = northInfo(false, { north: 'magnetic', yawSource: 'compass', yawStdDeg: 2 });
    expect(m).toMatchObject({ label: 'Magnetic north', tone: 'warn' });
    expect(m.note).toMatch(/^From the compass, which measures magnetic north \(±2°\)\. True north differs by the local declination/);
    expect(northInfo(false, { north: 'none', yawSource: 'none' })).toEqual({ label: 'Unknown', note: null, tone: 'neutral' });
  });
  it('older sessions without stats.north: compass yaw is MAGNETIC (webkitCompassHeading) unless a declination is reported', () => {
    expect(northInfo(true, { yawSource: 'gps' })).toMatchObject({ label: 'True north', tone: 'good' });
    expect(northInfo(true, { yawSource: 'compass' })).toMatchObject({ label: 'Magnetic north', tone: 'warn', note: expect.stringContaining('declination') });
    expect(northInfo(true, { yawSource: 'compass', declinationDeg: 2 }).label).toBe('True north');
    expect(northInfo(true, { yawSource: 'compass', northReference: 'true' }).label).toBe('True north');
    // a compass yaw reported with northAligned false is the magnetic-only yaw (applied, +Y ≈ magnetic north)
    expect(northInfo(false, { yawSource: 'compass' })).toMatchObject({ label: 'Magnetic north', tone: 'warn' });
    expect(northInfo(false, { yawSource: 'compass', declinationDeg: 3 }).label).toBe('Magnetic north');
    expect(northInfo(false, { yawSource: 'none' })).toEqual({ label: 'Unknown', note: null, tone: 'neutral' });
    expect(northInfo(false, {})).toEqual({ label: 'Unknown', note: null, tone: 'neutral' });
    expect(northInfo(undefined, {})).toMatchObject({ label: 'Unknown' });
    expect(northInfo(true, {})).toMatchObject({ label: 'True north', note: null });
  });
});

describe('memoryStatRows', () => {
  it('formats the engine memory statistics when present', () => {
    expect(memoryStatRows({})).toEqual([]);
    expect(memoryStatRows({ memoryMB: 412.4, denseMemPeakMB: 1180, denseMaxSize: 1360, denseMaxSizeRequested: 1600, fusionStep: 2 })).toEqual([
      ['Engine memory (last step)', '≈ 412 MB'],
      ['Dense peak (planned)', '≈ 1.18 GB'],
      ['Dense image size', '1360 px (1600 px requested — reduced to fit memory)'],
      ['Fusion pixel step', '2 (memory)'],
    ]);
    expect(memoryStatRows({ denseMaxSize: 1280, denseMaxSizeRequested: 1280, fusionStep: 1, memoryMB: 'n/a' })).toEqual([['Dense image size', '1280 px']]);
  });
});

describe('splat resume / progress', () => {
  it('engine checkpoint (loadSession) or an interrupted build of this app; needs a sparse model', () => {
    expect(splatResumable({ sparse, stats: { splatResumableIteration: 500 } })).toBe(true);
    expect(splatResumable({ sparse, stats: { splatResumableIteration: 0 } })).toBe(true);
    expect(splatResumable({ sparse, stats: { splatInterrupted: 1 } })).toBe(true);
    expect(splatResumable({ sparse, stats: {} })).toBe(false);
    expect(splatResumable({ sparse: null, stats: { splatResumableIteration: 500 } })).toBe(false);
  });
  it('resume quality: app marker first, then the engine’s; clear drops every marker', () => {
    expect(splatResumeQuality({ stats: { splatResumableQuality: 'max' } })).toBe('max');
    expect(splatResumeQuality({ stats: { splatInterruptedQuality: 'fast', splatResumableQuality: 'max' } })).toBe('fast');
    expect(splatResumeQuality({ stats: {} })).toBeNull();
    const st: Record<string, number | string> = { splatResumableIteration: 5, splatResumableQuality: 'max', splatInterrupted: 1, splatInterruptedQuality: 'fast', keyframes: 3 };
    clearSplatResume(st);
    expect(st).toEqual({ keyframes: 3 });
  });
  it('built quality of the splats', () => {
    expect(builtQuality({ quality: 'balanced', stats: { splatQuality: 'fast' } }, 'splat')).toBe('fast');
    expect(builtQuality({ quality: 'max', stats: {} }, 'splat')).toBe('max');
  });
  it('progress card: stage weights, iterations, speed and time left', () => {
    expect(splatProgressView({ stage: 'splat_prepare', progress: 0.5 })).toEqual({ progress: 0.04, text: 'Preparing the photos' });
    expect(splatProgressView({ stage: 'splat_prepare', progress: -1 }).progress).toBe(-1);
    const t = splatProgressView({ stage: 'splat_training', progress: 0.5, detail: { iteration: 1500, iterations: 3000, loss: 0.02, itPerSec: 25, gaussians: 1e5 } });
    expect(t.progress).toBeCloseTo(0.08 + 0.89 * 0.5, 9);
    expect(t.text).toBe('1,500 / 3,000 iterations · ~1 min left');
    expect(splatProgressView({ stage: 'splat_training', progress: 0.9, detail: { iteration: 2900, iterations: 3000, loss: 0, itPerSec: 6.5, gaussians: 1 } }).text).toBe(
      '2,900 / 3,000 iterations · < 1 min left',
    );
    expect(splatProgressView({ stage: 'splat_finishing', progress: 1 })).toEqual({ progress: 1, text: 'Saving the model' });
  });
  it('preview caption and "Use it now" (from finishFrom; disabled without a note while preparing)', () => {
    expect(splatPreviewCaption({ iteration: 0, iterations: 3000, viewId: 4, psnr: 19 })).toBe('Initial Gaussians, before training');
    expect(splatPreviewCaption({ iteration: 1250, iterations: 3000, viewId: 4, psnr: 27 })).toBe('Iteration 1,250 of 3,000');
    expect(splatPreviewCaption({ iteration: 3000, iterations: 3000 })).toBe('Iteration 3,000');
    expect(splatPreviewCaption()).toBe('Preview');
    expect(splatUseNow()).toEqual({ ready: false, note: '' });
    expect(splatUseNow({ iteration: 400, iterations: 3000, finishFrom: 900 })).toEqual({ ready: false, note: 'Usable from iteration 900' });
    expect(splatUseNow({ iteration: 900, iterations: 3000, finishFrom: 900 })).toEqual({ ready: true, note: '' });
    // an older engine without finishFrom: never offered
    expect(splatUseNow({ iteration: 2000, iterations: 3000 }).ready).toBe(false);
  });
  it('a model ended early stays continuable (not "interrupted") until a build of it is interrupted', () => {
    const splat = { count: 10, iterations: 1250, stats: { splatIterations: 3000 } } as unknown as SplatResult;
    expect(splatContinuable({ sparse, splat, stats: { splatResumableIteration: 1250 } })).toBe(true);
    // a continuation cancelled later (checkpoint further on) or here (app marker): interrupted
    expect(splatContinuable({ sparse, splat, stats: { splatResumableIteration: 1600 } })).toBe(false);
    expect(splatContinuable({ sparse, splat, stats: { splatResumableIteration: 1250, splatInterrupted: 1 } })).toBe(false);
    expect(splatContinuable({ sparse, splat: null, stats: { splatResumableIteration: 1250 } })).toBe(false);
    expect(splatContinuable({ sparse, splat, stats: {} })).toBe(false);
    expect(splatIterationsText(1250, 3000)).toBe('1,250 of 3,000 iterations');
    expect(splatIterationsText(3000, 3000)).toBe('3,000 iterations');
    expect(splatIterationsText(1250, undefined)).toBe('1,250 iterations');
  });
});
