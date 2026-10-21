import { describe, expect, it } from 'vitest';
import { ENGINE_ERRORS, namedError } from '../workers/protocol';
import { GUIDANCE_PRIORITY, GuidanceGate, frameErrorAction, isQuotaError, isStorageWriteError, overlayMatchesView } from './capture-policy';
import type { Guidance } from '../workers/protocol';

describe('frameErrorAction', () => {
  it('ignores back-pressure and cancellation', () => {
    expect(frameErrorAction(namedError(ENGINE_ERRORS.busy, 'busy'), 'ready')).toBe('ignore');
    expect(frameErrorAction(namedError(ENGINE_ERRORS.cancelled, 'x'), 'ready')).toBe('ignore');
  });
  it('a dead engine stops the capture instead of failing every frame at 10 Hz', () => {
    // EngineClient rejects every call with EngineCrashedError once the respawn cap is hit.
    const err = namedError(ENGINE_ERRORS.crashed, 'Engine stopped after repeated failures (gpu-lost)');
    expect(frameErrorAction(err, 'dead')).toBe('unavailable');
    expect(frameErrorAction(err, 'disposed')).toBe('unavailable');
    expect(frameErrorAction(new Error('whatever'), 'dead')).toBe('unavailable');
  });
  it('a respawn in progress keeps the capture going', () => {
    expect(frameErrorAction(namedError(ENGINE_ERRORS.crashed, 'Engine restarted: gpu-lost'), 'respawning')).toBe('restarted');
    expect(frameErrorAction(namedError(ENGINE_ERRORS.crashed, 'Engine restarted'), 'ready')).toBe('restarted');
  });
  it('missing / uninitialised engines are unavailable', () => {
    expect(frameErrorAction(namedError(ENGINE_ERRORS.notImplemented, 'engine not implemented yet'), 'idle')).toBe('unavailable');
    expect(frameErrorAction(namedError(ENGINE_ERRORS.notInitialized, 'init first'), 'ready')).toBe('unavailable');
  });
  it('failed writes pause the capture, quota or not', () => {
    expect(frameErrorAction(namedError('StorageQuotaError', 'Storage quota exceeded while storing a keyframe'), 'ready')).toBe('storage');
    expect(frameErrorAction(namedError('QuotaExceededError', 'The quota has been exceeded.'), 'ready')).toBe('storage');
    // an iOS UnknownError / closed connection, as the engine reports it: not a full disk
    const err = namedError(ENGINE_ERRORS.storageWrite, 'UnknownError: Connection to Indexed Database server lost', 'storage');
    expect(frameErrorAction(err, 'ready')).toBe('storage');
    expect(isQuotaError(err)).toBe(false);
    expect(isStorageWriteError(err)).toBe(true);
  });
  it('other errors are transient', () => {
    expect(frameErrorAction(new Error('decode failed'), 'ready')).toBe('transient');
    expect(frameErrorAction(namedError(ENGINE_ERRORS.timeout, 't'), 'ready')).toBe('transient');
    expect(frameErrorAction(null, 'ready')).toBe('transient');
  });
});

describe('overlayMatchesView', () => {
  it('draws only frames captured at the current UI rotation', () => {
    expect(overlayMatchesView(0, 0)).toBe(true);
    expect(overlayMatchesView(90, 90)).toBe(true);
    expect(overlayMatchesView(270, -90)).toBe(true);
    // A frame grabbed in portrait whose result arrives after the UI turned to landscape: its
    // keypoints are NOT a quarter turn of the new preview (the UI rotation compensates the roll).
    expect(overlayMatchesView(0, 90)).toBe(false);
    expect(overlayMatchesView(90, 270)).toBe(false);
    expect(overlayMatchesView(180, 0)).toBe(false);
  });
});

describe('GuidanceGate', () => {
  /** Feed one code per 100 ms frame (10 Hz) from t = `t0`; returns the shown code after each frame. */
  function feed(gate: GuidanceGate, codes: Guidance[], t0 = 0): (Guidance | null)[] {
    return codes.map((c, i) => gate.update(c, t0 + 100 * i).code);
  }

  it('shows the first code at once, a new one after 2 frames and 1.5 s of dwell', () => {
    const g = new GuidanceGate();
    expect(g.update('initializing', 0)).toEqual({ code: 'initializing', changed: true, announce: true });
    // 'good' is lower priority than 'initializing': confirmed after 2 frames, shown once 1.5 s passed
    const shown = feed(g, ['good', 'good', 'good', 'good', 'good', 'good', 'good', 'good', 'good', 'good', 'good', 'good', 'good', 'good', 'good', 'good'], 100);
    expect(shown.indexOf('good')).toBe(14); // t = 1500
    expect(g.code).toBe('good');
  });

  it('single-frame blips never reach the banner', () => {
    const g = new GuidanceGate();
    g.reset('good', 0);
    const shown = feed(g, ['move_slower', 'good', 'more_texture', 'good', 'translate_more', 'good', 'move_slower', 'good'], 2000);
    expect(shown.every((c) => c === 'good')).toBe(true);
    // alternating codes that each persist one frame neither
    expect(feed(g, ['move_slower', 'more_texture', 'move_slower', 'more_texture'], 3000).every((c) => c === 'good')).toBe(true);
  });

  it('a higher-priority code replaces a fresh one after its 2 frames; a lower one waits for the dwell', () => {
    const g = new GuidanceGate();
    g.reset('translate_more', 0);
    expect(g.update('move_slower', 100).changed).toBe(false);
    const r = g.update('move_slower', 200);
    expect(r).toEqual({ code: 'move_slower', changed: true, announce: false }); // both warnings: no announcement
    // back to 'good' (lowest): shown only 1.5 s after move_slower appeared
    const shown = feed(g, new Array<Guidance>(20).fill('good'), 300);
    expect(shown.indexOf('good')).toBe(14); // t = 1700 ≥ 200 + 1500
  });

  it("'tracking_lost' and 'too_dark' show on their first frame, even during the dwell", () => {
    const g = new GuidanceGate();
    g.reset('good', 0);
    expect(g.update('move_slower', 100).code).toBe('good');
    expect(g.update('move_slower', 200).code).toBe('move_slower');
    expect(g.update('tracking_lost', 300)).toEqual({ code: 'tracking_lost', changed: true, announce: true });
    expect(g.update('too_dark', 400)).toEqual({ code: 'too_dark', changed: true, announce: false }); // outranks lost, same severity
    // lost is lower than too_dark: it waits for the dwell like any other code
    expect(g.update('tracking_lost', 500).code).toBe('too_dark');
    expect(g.update('tracking_lost', 1900).code).toBe('tracking_lost');
  });

  it('announces severity changes only', () => {
    const g = new GuidanceGate({ dwellMs: 0 });
    const ann: boolean[] = [];
    let t = 0;
    for (const c of ['initializing', 'initializing', 'good', 'good', 'move_slower', 'move_slower', 'more_texture', 'more_texture', 'tracking_lost', 'good', 'good'] as Guidance[]) {
      const r = g.update(c, (t += 100));
      if (r.changed) ann.push(r.announce);
    }
    // initializing (first), good, move_slower, more_texture (same severity), tracking_lost, good
    expect(ann).toEqual([true, true, true, false, true, true]);
  });

  it('reset(null): the next code shows at once (resume after a pause)', () => {
    const g = new GuidanceGate();
    g.reset('move_slower', 0);
    g.reset();
    expect(g.code).toBeNull();
    expect(g.update('good', 50)).toEqual({ code: 'good', changed: true, announce: true });
  });

  it('ranks every guidance code', () => {
    const all: Guidance[] = ['good', 'move_slower', 'more_texture', 'translate_more', 'tracking_lost', 'initializing', 'too_dark'];
    expect([...GUIDANCE_PRIORITY].sort()).toEqual([...all].sort());
  });
});
