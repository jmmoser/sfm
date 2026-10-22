/**
 * Test data of the raw capture recordings (src/io/recording.ts): an in-memory RecordingSink, a
 * header, and FrameInputs with every FramePump field set, including the values JSON alone would
 * lose (−∞, −0, NaN, typed arrays). Used by recording.test.ts and the replay round-trip tests.
 */
import type { FrameInput } from '../workers/protocol';
import type { ImageStoreRef, RecordingHeader, RecordingSink } from './recording';

/** An in-memory RecordingSink (what the OPFS / IndexedDB sinks store, as plain maps). */
export class MemorySink implements RecordingSink {
  readonly kind = 'memory';
  header = '';
  readonly images = new Map<string, Uint8Array>();
  readonly lines: Record<string, string[]> = { 'frames.jsonl': [], 'live.jsonl': [] };
  flushes = 0;
  closed = false;
  failOn: 'putImage' | 'appendLine' | null = null;
  async writeHeader(json: string): Promise<void> {
    this.header = json;
  }
  async putImage(seq: number, jpeg: Uint8Array): Promise<ImageStoreRef> {
    if (this.failOn === 'putImage') throw Object.assign(new Error('The quota has been exceeded.'), { name: 'QuotaExceededError' });
    this.images.set(String(seq), jpeg.slice());
    return { key: String(seq) };
  }
  async appendLine(file: 'frames.jsonl' | 'live.jsonl', line: string): Promise<void> {
    if (this.failOn === 'appendLine') throw new Error('I/O error');
    this.lines[file].push(line);
  }
  async flush(): Promise<void> {
    this.flushes++;
  }
  async close(): Promise<void> {
    this.closed = true;
  }
  text(file: 'frames.jsonl' | 'live.jsonl'): string {
    return this.lines[file].map((l) => `${l}\n`).join('');
  }
}

export function header(over: Partial<RecordingHeader> = {}): RecordingHeader {
  return {
    format: 'sfm-raw-capture',
    version: 1,
    id: 'rec-test',
    name: 'Object · Oct 21, 14:02',
    createdAt: 1_790_000_000_000,
    timeOrigin: 1_789_999_990_000,
    app: { userAgent: 'test' },
    sessionId: 's1',
    engine: { mode: 'object', quality: 'balanced', hfovPriorDeg: 67.3, preferGPU: true },
    camera: { label: 'Back Camera', width: 1440, height: 1920 },
    capture: { grab: 'videoFrame', videoFrameOrientation: 'raw', targetHz: 10 },
    recorder: { maxSide: 1280, jpegQuality: 0.9, sink: 'memory', floorBytes: 1e9, maxPending: 3 },
    state: 'recording',
    ...over,
  };
}

/** A FrameInput with every field the FramePump sets (frame i of a sequence). */
export function pumpFrame(i: number): FrameInput {
  const t = 1000 + 100 * i + 0.125;
  return {
    bitmap: { width: 64, height: 48, data: new Uint8ClampedArray(64 * 48 * 4) },
    t,
    screenAngle: i < 3 ? 0 : 90,
    screenAngleSince: i < 3 ? -Infinity : 1234.5,
    imu: [
      { t: t - 60, gyro: [0.01 * i, -0, 0.3], accel: [0, 0, 0], accelG: [0.1, 9.8, -0.2] },
      { t: t - 10, gyro: [0.02, 0.01, NaN], accel: null, accelG: [0.1, 9.81, -0.2] },
    ],
    geo: i % 2 === 0 ? { t: t - 400, lat: 47.6, lon: -122.3, alt: 56.5, accuracy: 4.2, altitudeAccuracy: null } : null,
    heading: i === 1 ? null : { heading: 123.4 + i, accuracy: 12, north: 'magnetic', declinationDeg: null },
    index: i,
    segment: i < 5 ? 0 : 1,
    gravityDevice: Float64Array.of(0.01, -0.99, 0.05),
    attitude: { t: t - 5, R_world_device: Float64Array.of(1, 0, 0, 0, 0, -1, 0, 1, 0), absolute: i % 2 === 1, yawAccuracyDeg: i % 2 === 1 ? 20 : null },
    imuCalibrationVersion: 2,
    timeSource: i === 4 ? 'callback' : 'captureTime',
    rawBuffer: i >= 3,
    maxSide: i >= 3 ? 1280 : null,
  };
}

