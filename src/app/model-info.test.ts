import { describe, expect, it } from 'vitest';
import type { SparseResult } from '../workers/protocol';
import { denseResumable, memoryStatRows, northInfo } from './model-info';

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
    expect(northInfo(false, { yawSource: 'compass' })).toEqual({ label: 'Unknown', note: null, tone: 'neutral' });
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
