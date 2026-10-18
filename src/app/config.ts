/**
 * Stored capture options (localStorage `sfm.captureConfig.v1`, src/app/app.ts): parsing and
 * per-platform defaults, kept free of DOM imports so it runs in Node tests.
 */
import type { CaptureConfig } from './app';

/** Current version of the stored defaults (see CaptureConfig.configVersion). */
export const CONFIG_VERSION = 2;

/** Defaults for this platform (`apple` = iOS / iPadOS WebKit). */
export function defaultConfig(apple: boolean): CaptureConfig {
  return { mode: 'object', quality: 'balanced', useGps: false, torch: false, autoDense: true, debugOverlay: false, videoFrameGrab: apple, configVersion: CONFIG_VERSION };
}

/** Stored options (JSON or null) → a valid config; invalid / missing values take the defaults. */
export function parseConfig(raw: string | null, apple: boolean): CaptureConfig {
  const d = defaultConfig(apple);
  if (!raw) return d;
  let c: Partial<CaptureConfig>;
  try {
    c = JSON.parse(raw) as Partial<CaptureConfig>;
  } catch {
    return d;
  }
  if (!c || typeof c !== 'object') return d;
  const current = typeof c.configVersion === 'number' && c.configVersion >= CONFIG_VERSION;
  return {
    mode: c.mode === 'object' || c.mode === 'scene' || c.mode === 'auto' ? c.mode : d.mode,
    quality: c.quality === 'fast' || c.quality === 'balanced' || c.quality === 'max' ? c.quality : d.quality,
    useGps: typeof c.useGps === 'boolean' ? c.useGps : d.useGps,
    torch: false, // never start with the torch on
    autoDense: typeof c.autoDense === 'boolean' ? c.autoDense : d.autoDense,
    debugOverlay: typeof c.debugOverlay === 'boolean' ? c.debugOverlay : d.debugOverlay,
    // saved while the option was opt-in (default off): take the current default once
    videoFrameGrab: current && typeof c.videoFrameGrab === 'boolean' ? c.videoFrameGrab : d.videoFrameGrab,
    configVersion: CONFIG_VERSION,
  };
}

