import { afterEach, describe, expect, it } from 'vitest';
import { requestCapturePermissions } from './permissions';

const g = globalThis as Record<string, unknown>;
const nav = (globalThis as { navigator?: object }).navigator as Record<string, unknown> | undefined;

function install(name: string, value: unknown): void {
  Object.defineProperty(nav!, name, { value, configurable: true, writable: true });
}

afterEach(() => {
  delete g.DeviceMotionEvent;
  delete g.DeviceOrientationEvent;
  if (nav) {
    delete nav.mediaDevices;
    delete nav.geolocation;
  }
});

describe.runIf(!!nav)('requestCapturePermissions', () => {
  it('asks motion synchronously (inside the gesture), then orientation, camera, location', async () => {
    const calls: string[] = [];
    g.DeviceMotionEvent = { requestPermission: () => (calls.push('motion'), Promise.resolve('granted')) };
    g.DeviceOrientationEvent = { requestPermission: () => (calls.push('orientation'), Promise.resolve('granted')) };
    let stopped = 0;
    install('mediaDevices', {
      getUserMedia: async (c: MediaStreamConstraints) => {
        calls.push(`camera:${JSON.stringify(c.video)}`);
        return { getTracks: () => [{ stop: () => stopped++ }] };
      },
      enumerateDevices: async () => [{ kind: 'videoinput', label: 'Back Camera', deviceId: 'x' }],
    });
    install('geolocation', {
      getCurrentPosition: (ok: PositionCallback, _e: PositionErrorCallback, o: PositionOptions) => {
        calls.push(`geo:${o.enableHighAccuracy}`);
        ok({ timestamp: Date.now(), coords: { latitude: 1, longitude: 2, accuracy: 4, altitude: 10, altitudeAccuracy: 3 } } as unknown as GeolocationPosition);
      },
    });
    const p = requestCapturePermissions();
    expect(calls).toEqual(['motion']);
    const st = await p;
    expect(calls).toEqual(['motion', 'orientation', 'camera:{"facingMode":{"ideal":"environment"}}', 'geo:true']);
    expect(st.motion).toBe('granted');
    expect(st.camera).toBe('granted');
    expect(st.cameraLabels).toBe(true);
    expect(stopped).toBe(1);
    expect(st.geolocation).toBe('granted');
    expect(st.geoFix!.lat).toBe(1);
    expect(st.geoFix!.alt).toBe(10);
    expect(st.approximateLocation).toBe(false);
    expect(st.errors).toEqual([]);
  });

  it('reports a missing gesture, camera denial and location denial', async () => {
    g.DeviceMotionEvent = { requestPermission: () => Promise.reject(Object.assign(new Error('x'), { name: 'NotAllowedError' })) };
    install('mediaDevices', {
      getUserMedia: async () => {
        throw Object.assign(new Error('no'), { name: 'NotAllowedError' });
      },
      enumerateDevices: async () => [],
    });
    install('geolocation', {
      getCurrentPosition: (_ok: PositionCallback, e: PositionErrorCallback) => e({ code: 1, message: 'denied' } as GeolocationPositionError),
    });
    const st = await requestCapturePermissions();
    expect(st.motion).toBe('needs-gesture');
    expect(st.camera).toBe('denied');
    expect(st.geolocation).toBe('denied');
    expect(st.errors.length).toBe(2);
  });

  it('skips what is not requested; platforms without a motion permission model report granted', async () => {
    g.DeviceMotionEvent = function DeviceMotionEvent() {};
    const st = await requestCapturePermissions({ camera: false, geolocation: false });
    expect(st.motion).toBe('granted');
    expect(st.camera).toBe('not-requested');
    expect(st.geolocation).toBe('not-requested');
  });
});
