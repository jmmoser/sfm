import { describe, expect, it } from 'vitest';
import { CONFIG_VERSION, defaultConfig, parseConfig } from './config';

describe('parseConfig', () => {
  it('VideoFrame grab is on by default on iOS WebKit only', () => {
    expect(parseConfig(null, true).videoFrameGrab).toBe(true);
    expect(parseConfig(null, false).videoFrameGrab).toBe(false);
    expect(defaultConfig(true).configVersion).toBe(CONFIG_VERSION);
  });

  it('options saved while VideoFrame grab was opt-in take the new default once; later choices stick', () => {
    // saved after #13 (no configVersion): videoFrameGrab false was the old default
    const old = JSON.stringify({ mode: 'scene', quality: 'max', useGps: true, autoDense: false, debugOverlay: true, videoFrameGrab: false });
    const c = parseConfig(old, true);
    expect(c).toMatchObject({ mode: 'scene', quality: 'max', useGps: true, autoDense: false, debugOverlay: true, videoFrameGrab: true, configVersion: CONFIG_VERSION });
    // the user turns it off afterwards (saveConfig writes the current version): respected
    expect(parseConfig(JSON.stringify({ ...c, videoFrameGrab: false }), true).videoFrameGrab).toBe(false);
    // and on elsewhere when chosen explicitly
    expect(parseConfig(JSON.stringify({ ...defaultConfig(false), videoFrameGrab: true }), false).videoFrameGrab).toBe(true);
    // not on iOS: the old stored value meets the same default (off)
    expect(parseConfig(old, false).videoFrameGrab).toBe(false);
  });

  it('the cloud view (point cloud instead of the camera preview) is off by default and remembered', () => {
    expect(parseConfig(null, true).cloudView).toBe(false);
    expect(parseConfig(JSON.stringify({ ...defaultConfig(true), cloudView: true }), true).cloudView).toBe(true);
    expect(parseConfig(JSON.stringify({ ...defaultConfig(true), cloudView: 'yes' }), true).cloudView).toBe(false);
  });

  it('"Record raw capture" is off by default and remembered', () => {
    expect(parseConfig(null, true).recordRaw).toBe(false);
    expect(parseConfig(JSON.stringify({ ...defaultConfig(true), recordRaw: true }), true).recordRaw).toBe(true);
    expect(parseConfig(JSON.stringify({ ...defaultConfig(true), recordRaw: 1 }), true).recordRaw).toBe(false);
  });

  it('invalid values fall back to the defaults and the torch never starts on', () => {
    expect(parseConfig('{not json', true)).toEqual(defaultConfig(true));
    expect(parseConfig('null', false)).toEqual(defaultConfig(false));
    const c = parseConfig(JSON.stringify({ mode: 'x', quality: 7, torch: true, videoFrameGrab: 'yes', configVersion: CONFIG_VERSION }), true);
    expect(c).toMatchObject({ mode: 'object', quality: 'balanced', torch: false, videoFrameGrab: true });
  });
});
