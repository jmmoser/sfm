import { describe, expect, it } from 'vitest';
import { rotvecToMat3 } from '../math/rotation';
import {
  DEG_TO_RAD,
  MotionNormalizer,
  RAW_STRIDE,
  STANDARD_GRAVITY,
  defaultCalibration,
  deviceOrientationFromMatrix,
  deviceOrientationMatrix,
  deviceUpFromOrientation,
  normalizeIntervalMs,
  normalizeMotionEvent,
  rawToImuSample,
  type RawMotionEvent,
  type RawOrientationEvent,
} from './normalize';

const G = STANDARD_GRAVITY;
const RAD = 180 / Math.PI;

function mul(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const o = new Float64Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) for (let k = 0; k < 3; k++) o[3 * r + c] += a[3 * r + k] * b[3 * k + c];
  return o;
}
function rx(d: number): number[] {
  const r = d * DEG_TO_RAD;
  return [1, 0, 0, 0, Math.cos(r), -Math.sin(r), 0, Math.sin(r), Math.cos(r)];
}
function ry(d: number): number[] {
  const r = d * DEG_TO_RAD;
  return [Math.cos(r), 0, Math.sin(r), 0, 1, 0, -Math.sin(r), 0, Math.cos(r)];
}
function rz(d: number): number[] {
  const r = d * DEG_TO_RAD;
  return [Math.cos(r), -Math.sin(r), 0, Math.sin(r), Math.cos(r), 0, 0, 0, 1];
}

/** Deterministic tiny noise. */
function noise(i: number, k: number): number {
  const x = Math.sin(i * 12.9898 + k * 78.233) * 43758.5453;
  return (x - Math.floor(x) - 0.5) * 2;
}

function motionEvent(t: number, rr: [number, number, number], acc: [number, number, number] | null, accG: [number, number, number] | null, interval = 1 / 60): RawMotionEvent {
  return {
    timeStamp: t,
    interval,
    rotationRate: { alpha: rr[0], beta: rr[1], gamma: rr[2] },
    acceleration: acc ? { x: acc[0], y: acc[1], z: acc[2] } : null,
    accelerationIncludingGravity: accG ? { x: accG[0], y: accG[1], z: accG[2] } : null,
  };
}

function ori(t: number, a: number, b: number, g: number): RawOrientationEvent {
  return { timeStamp: t, alpha: a, beta: b, gamma: g };
}

/**
 * Simulate a still phone in a fixed pose: iOS sign (−1) or spec sign (+1) for the raw
 * accelerometer, with/without DeviceOrientation events. Returns the normaliser.
 */
function simulateStill(opts: {
  beta: number;
  gamma: number;
  platformSign: 1 | -1;
  prior: 1 | -1;
  withOrientation: boolean;
  theta?: number;
  n?: number;
}): MotionNormalizer {
  const norm = new MotionNormalizer({ accelSignPrior: opts.prior });
  norm.reset();
  const up = deviceUpFromOrientation(opts.beta, opts.gamma);
  const out = new Float64Array(RAW_STRIDE);
  const n = opts.n ?? 200;
  for (let i = 0; i < n; i++) {
    const t = 1000 + i * (1000 / 60);
    const f = [0, 1, 2].map((k) => opts.platformSign * (G * up[k] + 0.02 * noise(i, k))) as [number, number, number];
    const rr = [0, 1, 2].map((k) => 0.3 * noise(i, k + 3)) as [number, number, number]; // deg/s jitter
    norm.acceptMotion(motionEvent(t, rr, [0, 0, 0], f), opts.theta ?? 0, out, 0);
    if (opts.withOrientation) norm.pushOrientation(ori(t + 0.2, 30, opts.beta, opts.gamma));
  }
  return norm;
}

describe('units and axes', () => {
  it('interval: iOS seconds → ms, spec ms unchanged', () => {
    expect(normalizeIntervalMs(1 / 60)).toBeCloseTo(16.6667, 3);
    expect(normalizeIntervalMs(0.01666666753590107)).toBeCloseTo(16.6667, 3);
    expect(normalizeIntervalMs(16)).toBe(16);
    expect(normalizeIntervalMs(null)).toBeNaN();
    expect(normalizeIntervalMs(0)).toBeNaN();
  });

  it('rotationRate deg/s → rad/s with alpha = x, beta = y, gamma = z', () => {
    const s = normalizeMotionEvent(motionEvent(5, [RAD, 2 * RAD, -3 * RAD], null, [0, 0, -G]))!;
    expect(s.gyro[0]).toBeCloseTo(1, 12);
    expect(s.gyro[1]).toBeCloseTo(2, 12);
    expect(s.gyro[2]).toBeCloseTo(-3, 12);
    expect(s.t).toBe(5);
    expect(s.accel).toBeNull();
    expect(s.accelG).toEqual([0, 0, -G]); // default calibration = spec sign prior (+1)
    expect(normalizeMotionEvent({ timeStamp: 1, rotationRate: null })).toBeNull();
    expect(normalizeMotionEvent({ timeStamp: 1, rotationRate: { alpha: null, beta: 1, gamma: 1 } })).toBeNull();
  });

  it('applies the accelerometer signs of the calibration', () => {
    const cal = defaultCalibration(-1);
    const s = normalizeMotionEvent(motionEvent(0, [0, 0, 0], [1, 2, 3], [0, 0, -G]), cal)!;
    expect(s.accelG).toEqual([-0, -0, G]);
    expect(s.accel).toEqual([-1, -2, -3]);
  });
});

describe('DeviceOrientation Euler (Z-X′-Y″)', () => {
  it('matrix = Rz(α)·Rx(β)·Ry(γ) and round-trips within the spec ranges', () => {
    for (let i = 0; i < 200; i++) {
      const a = 360 * Math.abs(noise(i, 1));
      const b = -180 + 360 * (0.5 + 0.5 * noise(i, 2));
      const g = -89.9 + 179.8 * (0.5 + 0.5 * noise(i, 3));
      const R = deviceOrientationMatrix(a, b, g);
      const ref = mul(mul(rz(a), rx(b)), ry(g));
      for (let k = 0; k < 9; k++) expect(R[k]).toBeCloseTo(ref[k], 12);
      const e = deviceOrientationFromMatrix(R);
      const R2 = deviceOrientationMatrix(e.alpha, e.beta, e.gamma);
      for (let k = 0; k < 9; k++) expect(R2[k]).toBeCloseTo(R[k], 9);
      expect(e.alpha).toBeGreaterThanOrEqual(0);
      expect(e.alpha).toBeLessThan(360);
      expect(e.gamma).toBeGreaterThanOrEqual(-90);
      expect(e.gamma).toBeLessThan(90);
      expect(e.beta).toBeGreaterThanOrEqual(-180);
      expect(e.beta).toBeLessThan(180);
      const up = deviceUpFromOrientation(b, g);
      expect(up[0]).toBeCloseTo(R[6], 12);
      expect(up[1]).toBeCloseTo(R[7], 12);
      expect(up[2]).toBeCloseTo(R[8], 12);
    }
  });
  it('up vector for canonical poses', () => {
    const near = (v: ArrayLike<number>, w: number[]): void => w.forEach((x, i) => expect(v[i]).toBeCloseTo(x, 9));
    near(deviceUpFromOrientation(0, 0), [0, 0, 1]); // flat face-up
    near(deviceUpFromOrientation(90, 0), [0, 1, 0]); // upright portrait
    near(deviceUpFromOrientation(0, -90), [1, 0, 0]); // landscape, top edge left (right edge up)
    near(deviceUpFromOrientation(0, 90 - 1e-9), [-1, 0, 0]); // landscape, top edge right
  });
});

describe('MotionNormalizer: accelerometer sign self-check (check B)', () => {
  const poses: { name: string; beta: number; gamma: number; expected: [number, number, number] }[] = [
    { name: 'flat face-up', beta: 0, gamma: 0, expected: [0, 0, G] },
    { name: 'upright portrait', beta: 90, gamma: 0, expected: [0, G, 0] },
    { name: 'landscape (top edge left)', beta: 0, gamma: -90, expected: [G, 0, 0] },
    { name: 'tilted 40° back', beta: 50, gamma: 10, expected: Array.from(deviceUpFromOrientation(50, 10)).map((x) => G * x) as [number, number, number] },
  ];
  for (const p of poses) {
    it(`iOS (inverted) events, ${p.name}: detects −1 and outputs spec sign`, () => {
      // Deliberately wrong prior (+1) to prove the decision comes from the data.
      const norm = simulateStill({ beta: p.beta, gamma: p.gamma, platformSign: -1, prior: 1, withOrientation: true });
      expect(norm.calibration.accelSign).toBe(-1);
      expect(norm.calibration.accelSignSource).toBe('orientation');
      expect(norm.calibration.linearAccelSign).toBe(-1); // follows until measured
      const raw = new Float64Array(RAW_STRIDE);
      raw[6] = -p.expected[0];
      raw[7] = -p.expected[1];
      raw[8] = -p.expected[2];
      raw[9] = 2;
      const s = rawToImuSample(0, raw, 0, norm.calibration);
      for (let k = 0; k < 3; k++) expect(s.accelG![k]).toBeCloseTo(p.expected[k], 9);
    });
  }
  it('spec-sign platform (Chrome/Android) with an iOS prior: detects +1', () => {
    const norm = simulateStill({ beta: 70, gamma: -5, platformSign: 1, prior: -1, withOrientation: true });
    expect(norm.calibration.accelSign).toBe(1);
    expect(norm.calibration.accelSignSource).toBe('orientation');
  });
  it('does not decide while moving / inconsistent', () => {
    const norm = new MotionNormalizer({ accelSignPrior: 1 });
    norm.reset();
    const out = new Float64Array(RAW_STRIDE);
    for (let i = 0; i < 300; i++) {
      const t = i * 16.7;
      // spinning fast (gyro gate) — no votes
      norm.acceptMotion(motionEvent(t, [200, 0, 0], null, [0, 0, -G]), 0, out, 0);
      norm.pushOrientation(ori(t, 0, 0, 0));
    }
    expect(norm.calibration.accelSignSource).toBe('prior');
    expect(norm.stats.signVotesDown + norm.stats.signVotesUp).toBe(0);
  });
});

describe('MotionNormalizer: fallback UI-up heuristic (no orientation events)', () => {
  it('iOS upright portrait (θ=0) → −1 after ≥ 2 s', () => {
    const short = simulateStill({ beta: 90, gamma: 0, platformSign: -1, prior: 1, withOrientation: false, n: 100 });
    expect(short.calibration.accelSignSource).toBe('prior');
    const norm = simulateStill({ beta: 90, gamma: 0, platformSign: -1, prior: 1, withOrientation: false, n: 200 });
    expect(norm.calibration.accelSign).toBe(-1);
    expect(norm.calibration.accelSignSource).toBe('heuristic');
  });
  it('iOS landscape (θ=90, right edge up) → −1', () => {
    const norm = simulateStill({ beta: 0, gamma: -90, platformSign: -1, prior: 1, withOrientation: false, theta: 90 });
    expect(norm.calibration.accelSign).toBe(-1);
  });
  it('flat phone gives no heuristic votes', () => {
    const norm = simulateStill({ beta: 0, gamma: 0, platformSign: -1, prior: 1, withOrientation: false, n: 300 });
    expect(norm.calibration.accelSignSource).toBe('prior');
  });
});

describe('MotionNormalizer: acceptance', () => {
  it('drops the replayed first event after reset()', () => {
    const norm = new MotionNormalizer();
    norm.reset();
    const out = new Float64Array(RAW_STRIDE);
    expect(norm.acceptMotion(motionEvent(0, [1.234567, 0, 0], null, null), 0, out, 0)).toBe(false);
    expect(norm.acceptMotion(motionEvent(16, [1.234567, 0.1, 0], null, null), 0, out, 0)).toBe(true);
    expect(norm.stats.firstDropped).toBe(1);
  });
  it('drops bit-identical duplicates of unquantised data, keeps quantised repeats', () => {
    const norm = new MotionNormalizer({ dropFirst: false });
    const out = new Float64Array(RAW_STRIDE);
    const e = motionEvent(0, [0.123456789, -1.98765, 3.14159265], [0.01, 0.02, 0.03], [0.1, 0.2, -9.7]);
    expect(norm.acceptMotion(e, 0, out, 0)).toBe(true);
    expect(norm.acceptMotion({ ...e, timeStamp: 16 }, 0, out, 0)).toBe(false);
    expect(norm.stats.duplicates).toBe(1);
    const q = new MotionNormalizer({ dropFirst: false });
    let accepted = 0;
    for (let i = 0; i < 40; i++) if (q.acceptMotion(motionEvent(i * 16, [0.1, 0, -0.2], null, [0, 0, 9.8]), 0, out, 0)) accepted++;
    expect(accepted).toBeGreaterThanOrEqual(30); // after the quantisation evidence builds up, repeats are kept
  });
  it('rejects events without rotationRate', () => {
    const norm = new MotionNormalizer({ dropFirst: false });
    const out = new Float64Array(RAW_STRIDE);
    expect(norm.acceptMotion({ timeStamp: 0, rotationRate: null, accelerationIncludingGravity: { x: 0, y: 0, z: 9.8 } }, 0, out, 0)).toBe(false);
    expect(norm.stats.noGyro).toBe(1);
  });
});

describe('MotionNormalizer: linear acceleration sign (check B″)', () => {
  function run(linearPlatformSign: 1 | -1): MotionNormalizer {
    const norm = new MotionNormalizer({ accelSignPrior: 1 });
    norm.reset();
    const out = new Float64Array(RAW_STRIDE);
    const s = -1; // iOS accelerationIncludingGravity sign
    for (let i = 0; i < 600; i++) {
      const t = i * (1000 / 60);
      const ax = 3 * Math.sin(2 * Math.PI * 1.5 * (t / 1000));
      const ay = 1.5 * Math.cos(2 * Math.PI * 0.9 * (t / 1000));
      const f = [s * ax, s * ay, s * G] as [number, number, number]; // flat, accelerating in-plane
      const a = [linearPlatformSign * ax, linearPlatformSign * ay, 0] as [number, number, number];
      norm.acceptMotion(motionEvent(t, [0.01 * noise(i, 1), 0.01 * noise(i, 2), 0], a, f), 0, out, 0);
      norm.pushOrientation(ori(t + 0.1, 0, 0, 0));
    }
    return norm;
  }
  it('same sign as accelerationIncludingGravity (iOS) → linearAccelSign = accelSign = −1', () => {
    const n = run(-1);
    expect(n.calibration.accelSign).toBe(-1);
    expect(n.calibration.linearAccelSignSource).toBe('measured');
    expect(n.calibration.linearAccelSign).toBe(-1);
  });
  it('opposite sign → linearAccelSign = +1', () => {
    const n = run(1);
    expect(n.calibration.accelSign).toBe(-1);
    expect(n.calibration.linearAccelSign).toBe(1);
  });
});

describe('MotionNormalizer: gyro axes / units check (check D)', () => {
  /** Smooth body rate (rad/s) and the attitude obtained by fine integration. */
  function trajectory(n: number, dtMs: number): { t: number[]; R: Float64Array[]; w: number[][] } {
    const ts: number[] = [];
    const Rs: Float64Array[] = [];
    const ws: number[][] = [];
    let R = mul(rz(30), rx(70));
    const omega = (tt: number): number[] => [0.8 * Math.sin(2 * tt), -0.5 + 0.9 * Math.cos(3 * tt), 1.2 * Math.sin(1.3 * tt + 1)];
    let t = 0;
    const sub = 20;
    for (let i = 0; i < n; i++) {
      ts.push(1000 + t * 1000);
      Rs.push(R.slice());
      ws.push(omega(t));
      for (let k = 0; k < sub; k++) {
        const h = dtMs / 1000 / sub;
        const wm = omega(t + h / 2);
        R = mul(R, rotvecToMat3([wm[0] * h, wm[1] * h, wm[2] * h]));
        t += h;
      }
    }
    return { t: ts, R: Rs, w: ws };
  }
  function run(map: (w: number[]) => [number, number, number]): MotionNormalizer {
    const norm = new MotionNormalizer({ accelSignPrior: -1 });
    norm.reset();
    const out = new Float64Array(RAW_STRIDE);
    const tr = trajectory(400, 1000 / 60);
    for (let i = 0; i < tr.t.length; i++) {
      // Same CoreMotion tick: motion event first, then orientation (WebKit order).
      norm.acceptMotion(motionEvent(tr.t[i], map(tr.w[i]), null, null), 0, out, 0);
      const e = deviceOrientationFromMatrix(tr.R[i]);
      norm.pushOrientation(ori(tr.t[i] + 0.3, e.alpha, e.beta, e.gamma));
    }
    return norm;
  }
  function normalised(norm: MotionNormalizer, rr: [number, number, number]): number[] {
    const raw = new Float64Array(RAW_STRIDE);
    raw.set(rr);
    return rawToImuSample(0, raw, 0, norm.calibration).gyro;
  }
  it('verifies the spec mapping (alpha=x, beta=y, gamma=z, deg/s)', () => {
    const n = run((w) => [w[0] * RAD, w[1] * RAD, w[2] * RAD]);
    expect(n.calibration.gyroSource).toBe('verified');
    expect(n.stats.gyroCheck!.correlation).toBeGreaterThan(0.95);
    expect(n.stats.gyroCheck!.scale).toBeGreaterThan(0.9);
    expect(n.stats.gyroCheck!.scale).toBeLessThan(1.1);
  });
  it('corrects a permuted / sign-flipped mapping (alpha=z, beta=x, gamma=−y)', () => {
    const n = run((w) => [w[2] * RAD, w[0] * RAD, -w[1] * RAD]);
    expect(n.calibration.gyroSource).toBe('corrected');
    const g = normalised(n, [3 * RAD, 1 * RAD, -2 * RAD]); // raw for ω = (1, 2, 3)
    expect(g[0]).toBeCloseTo(1, 9);
    expect(g[1]).toBeCloseTo(2, 9);
    expect(g[2]).toBeCloseTo(3, 9);
  });
  it('corrects rad/s units (old Chrome)', () => {
    const n = run((w) => [w[0], w[1], w[2]]);
    expect(n.calibration.gyroSource).toBe('corrected');
    expect(n.calibration.gyroScale).toBe(1);
    const g = normalised(n, [0.5, -0.25, 1]);
    expect(g[0]).toBeCloseTo(0.5, 12);
    expect(g[1]).toBeCloseTo(-0.25, 12);
    expect(g[2]).toBeCloseTo(1, 12);
  });
  it('emits calibration changes with increasing version', () => {
    const norm = new MotionNormalizer({ accelSignPrior: 1 });
    const versions: number[] = [];
    norm.onChange((c) => versions.push(c.version));
    norm.setCalibration({ accelSign: -1 });
    expect(versions).toEqual([1]);
    expect(norm.calibration.accelSign).toBe(-1);
  });
});
