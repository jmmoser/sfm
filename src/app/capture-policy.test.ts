import { describe, expect, it } from 'vitest';
import { ENGINE_ERRORS, namedError } from '../workers/protocol';
import { frameErrorAction, isQuotaError, isStorageWriteError, overlayMatchesView } from './capture-policy';

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
