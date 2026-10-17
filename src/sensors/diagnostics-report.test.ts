import { describe, expect, it } from 'vitest';
import { diagnosticsReportJSON } from './diagnostics';

describe('diagnosticsReportJSON (Diagnostics "Copy")', () => {
  it('redacts camera device ids and any location, keeps the rest', () => {
    const report = {
      app: { mock: false },
      diagnostics: {
        cameras: [
          { deviceId: 'a1b2c3d4e5f6', groupId: 'g-77', label: 'Back Camera', lens: 'wide', facing: 'back' },
          { deviceId: '', label: '', lens: 'unknown', facing: 'unknown' },
        ],
        geo: { status: 'ok', fixes: 3, latest: { t: 1, lat: 47.3769, lon: 8.5417, alt: 408, accuracy: 5, altitudeAccuracy: null } },
        compass: { declination: { deg: 3.1, source: 'wmm', latDeg: 47.3769, lonDeg: 8.5417, altM: 408 } },
        motion: { calibration: { gyroMap: new Float64Array([1, 0, 0]) } },
        sparse: { metric: { originLatLonAlt: [47.3769, 8.5417, 408] } },
      },
    };
    const text = diagnosticsReportJSON(report);
    expect(text).not.toMatch(/a1b2c3d4e5f6|g-77|47\.37|8\.54|408/);
    const back = JSON.parse(text);
    expect(back.diagnostics.cameras[0]).toEqual({ deviceId: '[redacted]', groupId: '[redacted]', label: 'Back Camera', lens: 'wide', facing: 'back' });
    expect(back.diagnostics.cameras[1].deviceId).toBe(''); // "no id" stays visible
    expect(back.diagnostics.geo).toMatchObject({ status: 'ok', fixes: 3, latest: { lat: '[redacted]', lon: '[redacted]', alt: '[redacted]', accuracy: 5, altitudeAccuracy: null } });
    expect(back.diagnostics.compass.declination).toEqual({ deg: 3.1, source: 'wmm', latDeg: '[redacted]', lonDeg: '[redacted]', altM: '[redacted]' });
    expect(back.diagnostics.motion.calibration.gyroMap).toEqual([1, 0, 0]);
  });
});
