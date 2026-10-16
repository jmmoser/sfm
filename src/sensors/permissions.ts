/**
 * One-tap permission flow for capture (docs/research/ios-sensors.md §1.2, ios-platform.md §6):
 * motion + orientation FIRST (iOS requires the call synchronously inside a click/touchend
 * handler — `requestCapturePermissions` makes that call before returning), then the camera
 * (also yields device labels for lens selection), then geolocation (optional). Camera and
 * geolocation prompts do not need the gesture.
 */
import type { GeoFix } from '../core/types';
import { epochToPerfMs } from './geolocation';
import { MotionSensor } from './motion';

export type MotionPermissionState = 'granted' | 'denied' | 'unsupported' | 'needs-gesture' | 'not-requested';
export type CameraPermissionState = 'granted' | 'denied' | 'unavailable' | 'error' | 'not-requested';
export type GeoPermissionState = 'granted' | 'denied' | 'unsupported' | 'timeout' | 'not-requested';

export interface CapturePermissionStatus {
  secureContext: boolean;
  motion: MotionPermissionState;
  camera: CameraPermissionState;
  geolocation: GeoPermissionState;
  /** enumerateDevices() now returns labels (needed for lens classification). */
  cameraLabels: boolean;
  /** First fix obtained while asking (null if none). */
  geoFix: GeoFix | null;
  /** Location accuracy is coarse (iOS "Precise Location" off); null when unknown. */
  approximateLocation: boolean | null;
  /** The probe stream when `keepCameraStream` was set (caller must stop it). */
  stream: MediaStream | null;
  errors: string[];
}

export interface CapturePermissionOptions {
  motion?: boolean;
  camera?: boolean;
  geolocation?: boolean;
  /** Max wait for the first fix after the location prompt (ms, default 15000). */
  geolocationTimeoutMs?: number;
  /** Keep the camera probe stream open instead of stopping it (default false). */
  keepCameraStream?: boolean;
  facing?: 'back' | 'front';
}

function errName(e: unknown): string {
  return (e as { name?: string })?.name ?? String(e);
}

/**
 * Request everything capture needs from ONE user gesture. Call it synchronously at the top of
 * the tap handler (no await before it). Never rejects.
 */
export function requestCapturePermissions(opts: CapturePermissionOptions = {}): Promise<CapturePermissionStatus> {
  // Synchronous part — still inside the user gesture.
  const motionP: Promise<MotionPermissionState> =
    opts.motion === false
      ? Promise.resolve('not-requested')
      : MotionSensor.requestPermission().then((s) => (s === 'denied' && MotionSensor.lastPermissionError === 'NotAllowedError' ? 'needs-gesture' : s));
  return finishPermissions(motionP, opts);
}

async function finishPermissions(motionP: Promise<MotionPermissionState>, opts: CapturePermissionOptions): Promise<CapturePermissionStatus> {
  const status: CapturePermissionStatus = {
    secureContext: typeof isSecureContext === 'boolean' ? isSecureContext : true,
    motion: 'not-requested',
    camera: 'not-requested',
    geolocation: 'not-requested',
    cameraLabels: false,
    geoFix: null,
    approximateLocation: null,
    stream: null,
    errors: [],
  };
  try {
    status.motion = await motionP;
  } catch (e) {
    status.motion = 'denied';
    status.errors.push(`motion: ${errName(e)}`);
  }

  if (opts.camera !== false) {
    const md = typeof navigator !== 'undefined' ? navigator.mediaDevices : undefined;
    if (!md?.getUserMedia) {
      status.camera = 'unavailable';
      status.errors.push('camera: getUserMedia unavailable (HTTPS required)');
    } else {
      try {
        const s = await md.getUserMedia({
          video: { facingMode: { ideal: opts.facing === 'front' ? 'user' : 'environment' } },
          audio: false,
        });
        status.camera = 'granted';
        if (opts.keepCameraStream) status.stream = s;
        else s.getTracks().forEach((t) => t.stop());
        try {
          const devs = await md.enumerateDevices();
          status.cameraLabels = devs.some((d) => d.kind === 'videoinput' && !!d.label);
        } catch {
          /* ignore */
        }
      } catch (e) {
        const n = errName(e);
        status.camera = n === 'NotAllowedError' || n === 'SecurityError' ? 'denied' : n === 'NotFoundError' || n === 'OverconstrainedError' ? 'unavailable' : 'error';
        status.errors.push(`camera: ${n}`);
      }
    }
  }

  if (opts.geolocation !== false) {
    const geo = typeof navigator !== 'undefined' ? navigator.geolocation : undefined;
    if (!geo) status.geolocation = 'unsupported';
    else {
      const timeout = opts.geolocationTimeoutMs ?? 15000;
      const r = await new Promise<{ s: GeoPermissionState; p?: GeolocationPosition; err?: string }>((resolve) => {
        let settled = false;
        const guard = setTimeout(() => {
          if (!settled) {
            settled = true;
            resolve({ s: 'timeout' }); // prompt unanswered
          }
        }, timeout * 2);
        geo.getCurrentPosition(
          (p) => {
            if (settled) return;
            settled = true;
            clearTimeout(guard);
            resolve({ s: 'granted', p });
          },
          (e) => {
            if (settled) return;
            settled = true;
            clearTimeout(guard);
            // 2 = POSITION_UNAVAILABLE, 3 = TIMEOUT: permission was granted but no fix yet.
            resolve({ s: e.code === 1 ? 'denied' : 'granted', err: `geolocation: code ${e.code} ${e.message}` });
          },
          { enableHighAccuracy: true, maximumAge: 0, timeout },
        );
      });
      status.geolocation = r.s;
      if (r.err) status.errors.push(r.err);
      if (r.p) {
        const c = r.p.coords;
        status.geoFix = {
          t: epochToPerfMs(r.p.timestamp),
          lat: c.latitude,
          lon: c.longitude,
          alt: c.altitude ?? null,
          accuracy: c.accuracy,
          altitudeAccuracy: c.altitude == null ? null : (c.altitudeAccuracy ?? null),
        };
        status.approximateLocation = c.accuracy > 1000;
      }
    }
  }
  return status;
}

export type QueriedPermission = 'granted' | 'denied' | 'prompt' | 'unknown';

/** Non-prompting permission states via the Permissions API (motion is not queryable on iOS). */
export async function queryPermissionStates(): Promise<{ camera: QueriedPermission; geolocation: QueriedPermission }> {
  const q = async (name: string): Promise<QueriedPermission> => {
    try {
      const p = typeof navigator !== 'undefined' ? navigator.permissions : undefined;
      if (!p?.query) return 'unknown';
      const r = await p.query({ name } as PermissionDescriptor);
      return r.state as QueriedPermission;
    } catch {
      return 'unknown';
    }
  };
  const [camera, geolocation] = await Promise.all([q('camera'), q('geolocation')]);
  return { camera, geolocation };
}
