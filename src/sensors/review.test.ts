/**
 * Adversarial review tests for src/sensors (Node). Each block targets a suspected defect or a
 * realistic iOS edge case that the original tests did not exercise.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ImuSample } from '../core/types';
import { chooseCamera, classifyCamera, type CameraInfo } from './camera';
import { ScreenAngleTracker } from './frame-orientation';
import { FramePump, type PumpFrame } from './frame-pump';
import type { FrameCallback, FrameMeta } from './camera';
import { GeoSensor, type GeoPositionLike } from './geolocation';
import { ImuRingBuffer, MotionSensor } from './motion';
import { DEG_TO_RAD, MotionNormalizer, RAW_STRIDE, STANDARD_GRAVITY, azimuthDeg, deviceOrientationFromMatrix, deviceUpFromOrientation } from './normalize';
import { CompassSensor } from './orientation';
import { WakeLock } from './wakelock';

const RAD = 180 / Math.PI;

function mul(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const o = new Float64Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) for (let k = 0; k < 3; k++) o[3 * r + c] += a[3 * r + k] * b[3 * k + c];
  return o;
}
function rz(d: number): number[] {
  const r = d * DEG_TO_RAD;
  return [Math.cos(r), -Math.sin(r), 0, Math.sin(r), Math.cos(r), 0, 0, 0, 1];
}
function rx(d: number): number[] {
  const r = d * DEG_TO_RAD;
  return [1, 0, 0, 0, Math.cos(r), -Math.sin(r), 0, Math.sin(r), Math.cos(r)];
}
function noise(i: number, k: number): number {
  const x = Math.sin(i * 12.9898 + k * 78.233) * 43758.5453;
  return (x - Math.floor(x) - 0.5) * 2;
}
function angDiff(a: number, b: number): number {
  let d = (a - b) % 360;
  if (d > 180) d -= 360;
  if (d < -180) d += 360;
  return Math.abs(d);
}
function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

// -------------------------------------------------------------------------------------------
// IMU ring buffer: burst re-timing must keep sample times monotonic (binary searches rely on it)
// -------------------------------------------------------------------------------------------
describe('ImuRingBuffer invariants under random bursts, clock steps back and drains', () => {
  it('times stay non-decreasing, never later than arrival, frozen samples never move', () => {
    for (let seed = 1; seed <= 20; seed++) {
      const b = new ImuRingBuffer(64);
      b.setNominal(1000 / 60);
      const rec = new Float64Array(RAW_STRIDE);
      let t = 1000;
      const frozenTimes = new Map<number, number>();
      for (let i = 0; i < 600; i++) {
        const r = (noise(i, seed) + 1) / 2;
        if (r < 0.15) t += 0.05 + 0.3 * r; // burst (back-to-back dispatch after a stall)
        else if (r < 0.18) t -= 2; // non-monotonic timeStamp
        else if (r < 0.2) t += 150; // gap
        else t += 1000 / 60 + 3 * noise(i, seed + 7);
        b.push(t, rec, 0);
        if (i % 37 === 0) {
          // drain: record the handed-out times
          for (let k = b.first; k < b.end; k++) if (!frozenTimes.has(k)) frozenTimes.set(k, b.timeAt(k));
          b.freeze(b.end);
        }
        for (let k = b.first + 1; k < b.end; k++) expect(b.timeAt(k)).toBeGreaterThanOrEqual(b.timeAt(k - 1));
        for (let k = b.first; k < b.end; k++) expect(b.timeAt(k)).toBeLessThanOrEqual(b.arrivalAt(k));
        for (const [k, v] of frozenTimes) if (k >= b.first) expect(b.timeAt(k)).toBe(v);
      }
    }
  });
});

// -------------------------------------------------------------------------------------------
// Gyro self-check D in the REAL capture pose: phone upright (β ≈ 90°, Euler gimbal lock region),
// panning about the vertical, iOS timestamps floored to 1 ms with dispatch jitter.
// -------------------------------------------------------------------------------------------
describe('MotionNormalizer check D near gimbal lock (upright capture pose)', () => {
  function run(map: (w: number[]) => [number, number, number]): MotionNormalizer {
    const norm = new MotionNormalizer({ accelSignPrior: -1 });
    norm.reset();
    const out = new Float64Array(RAW_STRIDE);
    // R_wd(t) = Rz(yaw(t)) · Rx(90° + pitch(t)) · Rz(roll(t)) (upright, panning, small tilt/roll)
    const pose = (s: number): Float64Array =>
      mul(mul(rz(40 * Math.sin(0.9 * s) + 25 * s), rx(90 + 6 * Math.sin(1.7 * s))), rz(4 * Math.sin(2.3 * s)));
    // body rate by central difference on the analytic attitude
    const rate = (s: number): number[] => {
      const h = 1e-5;
      const d = mul(
        Float64Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8].map((k) => pose(s - h)[[0, 3, 6, 1, 4, 7, 2, 5, 8][k]])),
        pose(s + h),
      );
      // log of a small rotation
      return [(d[7] - d[5]) / (4 * h), (d[2] - d[6]) / (4 * h), (d[3] - d[1]) / (4 * h)];
    };
    // first event replayed/dropped
    for (let i = 0; i < 600; i++) {
      const s = i / 60;
      const tTrue = 1000 + s * 1000;
      const tMotion = Math.floor(tTrue + 1.5 + 2 * Math.abs(noise(i, 2))); // web-process creation time, 1 ms floor
      const w = rate(s);
      norm.acceptMotion(
        { timeStamp: tMotion, interval: 1 / 60, rotationRate: { alpha: map(w)[0], beta: map(w)[1], gamma: map(w)[2] }, acceleration: null, accelerationIncludingGravity: null },
        0,
        out,
        0,
      );
      const e = deviceOrientationFromMatrix(pose(s));
      norm.pushOrientation({ timeStamp: tMotion + (noise(i, 4) > 0.6 ? 1 : 0), alpha: e.alpha, beta: e.beta, gamma: e.gamma });
    }
    return norm;
  }
  it('verifies (does not "correct") the spec mapping', () => {
    const n = run((w) => [w[0] * RAD, w[1] * RAD, w[2] * RAD]);
    expect(n.stats.gyroCheck).not.toBeNull();
    expect(n.calibration.gyroSource).toBe('verified');
    expect(Array.from(n.calibration.gyroMap)).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    expect(n.calibration.gyroScale).toBe(DEG_TO_RAD);
  });
  it('still detects the legacy "alpha = z" mapping in that pose', () => {
    const n = run((w) => [w[2] * RAD, w[0] * RAD, w[1] * RAD]);
    expect(n.calibration.gyroSource).toBe('corrected');
    expect(Array.from(n.calibration.gyroMap)).toEqual([0, 1, 0, 0, 0, 1, 1, 0, 0]);
  });
});

// -------------------------------------------------------------------------------------------
// Gyro self-check D during a pure pan (rotation about device y only, phone upright): the x/z
// rows of the correlation matrix are noise, so the mapping of those axes is undeterminable.
// -------------------------------------------------------------------------------------------
describe('MotionNormalizer check D with single-axis motion (panning)', () => {
  function pan(norm: MotionNormalizer, map: (w: number[]) => number[], seed: number, i0: number, n: number, extra: (s: number) => Float64Array | null): void {
    const out = new Float64Array(RAW_STRIDE);
    for (let i = i0; i < i0 + n; i++) {
      const s = i / 60;
      const t = 1000 + s * 1000;
      const R = extra(s) ?? mul(rz(46 * s), rx(90)); // 0.8 rad/s pan, phone upright
      // body rate of R = Rz(ẏ s)·Rx(90): (0, ẏ, 0) + gyro noise (σ ≈ 0.3 °/s)
      const w = extra(s) ? bodyRate(extra, s) : [0, 46, 0];
      const raw = map([w[0] + 0.5 * noise(i, seed), w[1] + 0.5 * noise(i, seed + 50), w[2] + 0.5 * noise(i, seed + 99)]);
      norm.acceptMotion({ timeStamp: t, interval: 1 / 60, rotationRate: { alpha: raw[0], beta: raw[1], gamma: raw[2] }, acceleration: null, accelerationIncludingGravity: null }, 0, out, 0);
      const e = deviceOrientationFromMatrix(R);
      norm.pushOrientation({ timeStamp: t + 0.3, alpha: e.alpha, beta: e.beta + 0.05 * noise(i, seed + 7), gamma: e.gamma });
    }
  }
  function bodyRate(f: (s: number) => Float64Array | null, s: number): number[] {
    const h = 1e-5;
    const a = f(s - h)!;
    const b = f(s + h)!;
    const at = [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
    const d = mul(at, b);
    return [((d[7] - d[5]) / (4 * h)) * RAD, ((d[2] - d[6]) / (4 * h)) * RAD, ((d[3] - d[1]) / (4 * h)) * RAD];
  }
  it('never "corrects" the mapping from a pan (was a random signed x/z permutation in ~90 % of runs)', () => {
    for (let seed = 0; seed < 40; seed++) {
      const norm = new MotionNormalizer({ accelSignPrior: -1 });
      norm.reset();
      pan(norm, (w) => w, seed, 0, 400, () => null);
      expect(norm.calibration.gyroSource).not.toBe('corrected');
      expect(Array.from(norm.calibration.gyroMap)).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    }
  });
  it('decides once the motion covers every axis (pan first, then free hand motion)', () => {
    const free = (s: number): Float64Array => mul(mul(rz(46 * s + 20 * Math.sin(1.1 * s)), rx(90 + 25 * Math.sin(1.9 * s))), rz(20 * Math.sin(2.3 * s + 1)));
    for (const [name, map, expected] of [
      ['spec', (w: number[]) => w, 'verified'],
      ['alpha=z legacy', (w: number[]) => [w[2], w[0], w[1]], 'corrected'],
    ] as const) {
      const norm = new MotionNormalizer({ accelSignPrior: -1 });
      norm.reset();
      pan(norm, map, 3, 0, 300, () => null);
      expect(norm.calibration.gyroSource, name).toBe('default');
      pan(norm, map, 3, 300, 600, free);
      expect(norm.calibration.gyroSource, name).toBe(expected);
      if (expected === 'corrected') expect(Array.from(norm.calibration.gyroMap)).toEqual([0, 1, 0, 0, 0, 1, 1, 0, 0]);
    }
  });
});

// -------------------------------------------------------------------------------------------
// Accelerometer sign check B on an iOS stream that STARTS with the replayed (stale) event and
// includes duplicates, landscape UI, phone held upright in landscape.
// -------------------------------------------------------------------------------------------
describe('MotionSensor end-to-end on a simulated iOS stream (landscape, upright)', () => {
  it('detects −1 from orientation pairing, outputs spec-sign accelG ≈ +up and gravityAt ≈ down', () => {
    const target = new EventTarget();
    const m = new MotionSensor({ target, accelSignPrior: -1, screenAngle: () => 90 });
    // landscape, top edge LEFT (θ = 90): device +x points up; slight forward tilt
    const Rwd = mul(rz(10), mul(rx(80), rz(90))); // Rz(90): device +x → portrait "up" axis
    const e = deviceOrientationFromMatrix(Rwd);
    const up = deviceUpFromOrientation(e.beta, e.gamma);
    expect(up[0]).toBeGreaterThan(0.9); // right edge up (sanity of the synthetic pose)
    for (let i = 0; i < 120; i++) {
      const t = 5000 + i * (1000 / 60);
      const iosRaw = [0, 1, 2].map((k) => -(STANDARD_GRAVITY * up[k] + 0.02 * noise(i, k)));
      const ev = { timeStamp: t, interval: 1 / 60, rotationRate: { alpha: 0.2 * noise(i, 5), beta: 0.2 * noise(i, 6), gamma: 0.1 }, acceleration: { x: 0, y: 0, z: 0 }, accelerationIncludingGravity: { x: iosRaw[0], y: iosRaw[1], z: iosRaw[2] } };
      m.handleMotion(ev);
      if (i % 10 === 3) m.handleMotion({ ...ev, timeStamp: t + 1 }); // duplicate CoreMotion poll
      m.handleOrientation({ timeStamp: t + 0.3, alpha: e.alpha, beta: e.beta, gamma: e.gamma });
    }
    expect(m.calibration.accelSign).toBe(-1);
    expect(m.calibration.accelSignSource).toBe('orientation');
    expect(m.stats.duplicates).toBeGreaterThan(5);
    const s = m.samplesBetween(5000, 7000);
    const a = s[s.length - 1].accelG!;
    expect(a[0] * up[0] + a[1] * up[1] + a[2] * up[2]).toBeGreaterThan(9.5); // specific force points UP
    const g = m.gravityAt(6000)!;
    expect(g[0] * up[0] + g[1] * up[1] + g[2] * up[2]).toBeLessThan(-0.999); // gravity points DOWN
  });
});

// -------------------------------------------------------------------------------------------
// Compass: iOS resets the arbitrary yaw reference of DeviceOrientation alpha whenever motion
// updates restart (page hidden → visible). ψ must re-converge quickly, not over ~τ = 30 s.
// -------------------------------------------------------------------------------------------
describe('CompassSensor after a CoreMotion restart (new arbitrary yaw reference)', () => {
  it('re-estimates ψ within ~1 s of new data after a gap in the event stream', () => {
    const c = new CompassSensor({ target: null, screenAngle: () => 0 });
    const feed = (psi: number, t0: number, n: number): void => {
      for (let i = 0; i < n; i++) {
        const t = t0 + i * (1000 / 60);
        // tilted phone (top near horizontal) slowly yawing
        const Renu = mul(rz(20 + 10 * Math.sin(i / 50)), rx(30));
        const Rarb = mul(rz(psi), Renu);
        const e = deviceOrientationFromMatrix(Rarb);
        const heading = azimuthDeg(Renu[1], Renu[4]);
        c.handleOrientation({ timeStamp: t, alpha: e.alpha, beta: e.beta, gamma: e.gamma, webkitCompassHeading: (heading + noise(i, 1) + 360) % 360, webkitCompassAccuracy: 10 });
      }
    };
    feed(40, 1000, 1800); // 30 s with reference ψ = 40
    expect(angDiff(c.yawOffsetDeg!, 40)).toBeLessThan(2);
    // page hidden for 5 s; CoreMotion restarts with a new arbitrary yaw: ψ = 130
    feed(130, 1000 + 30_000 + 5000, 60); // 1 s of new data
    expect(c.yawOffsetDeg).not.toBeNull();
    expect(angDiff(c.yawOffsetDeg!, 130)).toBeLessThan(3);
  });
});

// -------------------------------------------------------------------------------------------
// FramePump lifecycle
// -------------------------------------------------------------------------------------------
class FakeBitmap {
  closed = false;
  constructor(
    readonly width: number,
    readonly height: number,
  ) {}
  close(): void {
    this.closed = true;
  }
}
class SlowCamera {
  cbs = new Set<FrameCallback>();
  grabs: FakeBitmap[] = [];
  pending: (() => void)[] = [];
  fail = false;
  calls = 0;
  meta: FrameMeta = { mediaTime: 0, width: 640, height: 480, callbackTime: 0, timeSource: 'captureTime', segment: 0 };
  onFrame(cb: FrameCallback): () => void {
    this.cbs.add(cb);
    return () => this.cbs.delete(cb);
  }
  grabFrame(): Promise<ImageBitmap> {
    this.calls++;
    if (this.fail) return Promise.reject(new Error('createImageBitmap failed'));
    const b = new FakeBitmap(640, 480);
    this.grabs.push(b);
    return new Promise((res) => this.pending.push(() => res(b as unknown as ImageBitmap)));
  }
  emit(t: number): void {
    for (const cb of this.cbs) cb(t, this.meta);
  }
}

describe('FramePump lifecycle (adversarial)', () => {
  it('does not deliver a frame whose grab completes after stop(); the bitmap is closed', async () => {
    const cam = new SlowCamera();
    const frames: PumpFrame[] = [];
    let drained = 0;
    const pump = new FramePump({
      camera: cam,
      angleTracker: new ScreenAngleTracker({ read: () => 0 }),
      isHidden: () => false,
      motion: {
        drain: (): ImuSample[] => {
          drained++;
          return [];
        },
      },
      consumer: (f) => void frames.push(f),
    });
    pump.start();
    cam.emit(0);
    expect(cam.grabs.length).toBe(1);
    pump.stop();
    cam.pending.shift()!(); // grab resolves after stop()
    await sleep(0);
    expect(frames.length).toBe(0);
    expect(cam.grabs[0].closed).toBe(true);
    expect(drained).toBe(0); // IMU samples stay buffered for the next session
    expect(pump.busy).toBe(false);
  });

  it('rate-limits retries when grabbing fails (no 30 Hz error storm)', async () => {
    const cam = new SlowCamera();
    cam.fail = true;
    const errors: unknown[] = [];
    const pump = new FramePump({
      camera: cam,
      angleTracker: new ScreenAngleTracker({ read: () => 0 }),
      isHidden: () => false,
      targetHz: 10,
      onError: (e) => errors.push(e),
      consumer: () => undefined,
    });
    pump.start();
    for (let i = 0; i < 30; i++) {
      cam.emit(i * (1000 / 30));
      await sleep(0);
    }
    pump.stop();
    expect(errors.length).toBeGreaterThan(0);
    expect(cam.calls).toBeLessThanOrEqual(11); // ≈ targetHz over 1 s, not every frame
  });
});

// -------------------------------------------------------------------------------------------
// GeoSensor: the first iOS fix is often coarse (cell / Wi-Fi) before GNSS converges.
// -------------------------------------------------------------------------------------------
describe('GeoSensor.approximate reflects the current precision', () => {
  it('is not stuck at true after a coarse first fix', () => {
    const g = new GeoSensor({ geolocation: null });
    const pos = (t: number, acc: number): GeoPositionLike => ({ timestamp: t, coords: { latitude: 47, longitude: 8, accuracy: acc } });
    g.handlePosition(pos(1000, 1414), 2000, 2000);
    expect(g.approximate).toBe(true);
    g.handlePosition(pos(2000, 12), 3000, 3000);
    g.handlePosition(pos(3000, 6), 4000, 4000);
    expect(g.approximate).toBe(false);
  });
});

// -------------------------------------------------------------------------------------------
// Camera choice in a locale whose labels the keyword lists do not cover (e.g. Thai, Greek):
// only capabilities are usable. WebKit enumerates UltraWide BEFORE Wide.
// -------------------------------------------------------------------------------------------
describe('chooseCamera with unrecognised labels (capabilities only)', () => {
  const env = ['environment'];
  let n = 0;
  const cam = (id: string, caps: Record<string, unknown>): CameraInfo => {
    const label = `กล้อง ${++n}`; // Thai-like label: no keyword matches (ids are not part of it)
    return { deviceId: id, label, ...classifyCamera(label, caps), capabilities: caps };
  };
  it('Pro iPhone: picks the physical wide camera (median minimum focus distance), not the ultra-wide', () => {
    const cams = [
      cam('front', { facingMode: ['user'], zoom: { min: 1, max: 10 } }),
      cam('triple', { facingMode: env, zoom: { min: 0.5, max: 10 }, focusDistance: { min: 0.02 } }),
      cam('dualwide', { facingMode: env, zoom: { min: 0.5, max: 10 }, focusDistance: { min: 0.02 } }),
      cam('ultrawide', { facingMode: env, zoom: { min: 1, max: 10 }, focusDistance: { min: 0.02 } }),
      cam('dual', { facingMode: env, zoom: { min: 1, max: 10 }, focusDistance: { min: 0.15 } }),
      cam('wide', { facingMode: env, zoom: { min: 1, max: 10 }, focusDistance: { min: 0.15 } }),
      cam('tele', { facingMode: env, zoom: { min: 1, max: 10 }, focusDistance: { min: 0.5 } }),
    ];
    expect(chooseCamera(cams)!.deviceId).toBe('wide');
  });
  it('dual-lens iPhone: wide (longer minimum focus distance) over ultra-wide', () => {
    const cams = [
      cam('front', { facingMode: ['user'] }),
      cam('dualwide', { facingMode: env, zoom: { min: 0.5, max: 10 }, focusDistance: { min: 0.02 } }),
      cam('ultrawide', { facingMode: env, zoom: { min: 1, max: 10 }, focusDistance: { min: 0.02 } }),
      cam('wide', { facingMode: env, zoom: { min: 1, max: 10 }, focusDistance: { min: 0.12 } }),
    ];
    expect(chooseCamera(cams)!.deviceId).toBe('wide');
  });
  it('keeps enumeration order when no focus distances are reported (Android / desktop)', () => {
    const cams = [cam('a', { facingMode: env }), cam('b', { facingMode: env })];
    expect(chooseCamera(cams)!.deviceId).toBe('a');
  });
});

// -------------------------------------------------------------------------------------------
// WakeLock: on iOS the 'release' event of the old sentinel can be dispatched only after the page
// is visible again (tasks are suspended while hidden) — i.e. after the lock was re-acquired.
// -------------------------------------------------------------------------------------------
class FakeSentinel extends EventTarget {
  released = false;
  readonly type = 'screen';
  async release(): Promise<void> {
    if (this.released) return;
    this.released = true;
    this.dispatchEvent(new Event('release'));
  }
}
describe('WakeLock re-acquisition', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });
  it('a late release event of the previous sentinel does not orphan the current lock', async () => {
    const sentinels: FakeSentinel[] = [];
    vi.stubGlobal('navigator', {
      wakeLock: {
        request: async (): Promise<FakeSentinel> => {
          const s = new FakeSentinel();
          sentinels.push(s);
          return s;
        },
      },
    });
    const doc = new EventTarget() as EventTarget & { visibilityState: string; hidden: boolean };
    doc.visibilityState = 'visible';
    doc.hidden = false;
    vi.stubGlobal('document', doc);

    const wl = new WakeLock();
    await wl.acquire();
    expect(sentinels.length).toBe(1);
    expect(wl.active).toBe(true);
    // page hidden: the system releases the lock, but the 'release' event is not dispatched yet
    doc.visibilityState = 'hidden';
    doc.hidden = true;
    sentinels[0].released = true;
    doc.dispatchEvent(new Event('visibilitychange'));
    expect(sentinels.length).toBe(1);
    // visible again: re-acquired
    doc.visibilityState = 'visible';
    doc.hidden = false;
    doc.dispatchEvent(new Event('visibilitychange'));
    await sleep(0);
    expect(sentinels.length).toBe(2);
    expect(wl.active).toBe(true);
    // now the stale 'release' event of the first sentinel arrives
    sentinels[0].dispatchEvent(new Event('release'));
    await sleep(0);
    expect(wl.active).toBe(true);
    await wl.release();
    // every sentinel ever obtained must be released (no orphaned screen lock)
    expect(sentinels.every((s) => s.released)).toBe(true);
    expect(wl.active).toBe(false);
  });
});

