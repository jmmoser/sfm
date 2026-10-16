import { describe, expect, it } from 'vitest';
import type { ImuSample } from '../core/types';
import { rotvecToMat3 } from '../math/rotation';
import { ImuRingBuffer, MotionSensor, angularSpeedAt, integrateGyro } from './motion';
import { RAW_STRIDE, STANDARD_GRAVITY, deviceUpFromOrientation, type RawMotionEvent } from './normalize';

const RAD = 180 / Math.PI;
const G = STANDARD_GRAVITY;

function mul(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const o = new Float64Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) for (let k = 0; k < 3; k++) o[3 * r + c] += a[3 * r + k] * b[3 * k + c];
  return o;
}
function tr(a: ArrayLike<number>): Float64Array {
  return new Float64Array([a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]);
}
function angleBetween(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const d = mul(tr(a), b);
  return Math.acos(Math.min(1, Math.max(-1, 0.5 * (d[0] + d[4] + d[8] - 1))));
}
function rz(r: number): number[] {
  return [Math.cos(r), -Math.sin(r), 0, Math.sin(r), Math.cos(r), 0, 0, 0, 1];
}
function rx(r: number): number[] {
  return [1, 0, 0, 0, Math.cos(r), -Math.sin(r), 0, Math.sin(r), Math.cos(r)];
}
function noise(i: number, k: number): number {
  const x = Math.sin(i * 12.9898 + k * 78.233) * 43758.5453;
  return (x - Math.floor(x) - 0.5) * 2;
}

/** Analytic attitude R_wd(t) = Rz(a(t))·Rx(b(t)) and its body rate ω = (ḃ, ȧ sin b, ȧ cos b). */
const A = 0.6,
  B = 0.5,
  F1 = 0.9,
  F2 = 1.3;
function att(ts: number): Float64Array {
  const t = ts / 1000;
  return mul(rz(A * Math.sin(2 * Math.PI * F1 * t)), rx(B * Math.sin(2 * Math.PI * F2 * t + 0.4)));
}
function omega(ts: number): [number, number, number] {
  const t = ts / 1000;
  const b = B * Math.sin(2 * Math.PI * F2 * t + 0.4);
  const ad = A * 2 * Math.PI * F1 * Math.cos(2 * Math.PI * F1 * t);
  const bd = B * 2 * Math.PI * F2 * Math.cos(2 * Math.PI * F2 * t + 0.4);
  return [bd, ad * Math.sin(b), ad * Math.cos(b)];
}
function samplesAt(times: number[]): ImuSample[] {
  return times.map((t) => ({ t, gyro: omega(t), accel: null, accelG: null }));
}

describe('integrateGyro', () => {
  it('is exact for a constant rate (off-grid interval ends)', () => {
    const w: [number, number, number] = [0.3, -0.2, 0.5];
    const s: ImuSample[] = [];
    for (let i = 0; i < 100; i++) s.push({ t: i * 16.6667, gyro: w, accel: null, accelG: null });
    const t0 = 123.4;
    const t1 = 987.6;
    const dt = (t1 - t0) / 1000;
    const R = integrateGyro(s, t0, t1);
    const expected = rotvecToMat3([-w[0] * dt, -w[1] * dt, -w[2] * dt]); // R_wd(t1)ᵀ R_wd(t0) = exp(−ω dt)
    expect(angleBetween(R, expected)).toBeLessThan(1e-12);
    // extrapolation beyond the samples holds ω constant
    const R2 = integrateGyro(s, 1600, 1800);
    expect(angleBetween(R2, rotvecToMat3([-w[0] * 0.2, -w[1] * 0.2, -w[2] * 0.2]))).toBeLessThan(1e-12);
  });

  it('matches an analytic rotation (60 Hz, jittered timestamps) to < 3e-5 rad over ≤ 2.5 s', () => {
    const times: number[] = [];
    for (let i = 0; i < 240; i++) times.push(i * (1000 / 60) + 2 * noise(i, 1));
    times.sort((a, b) => a - b);
    const s = samplesAt(times);
    let worst = 0;
    let worstShort = 0;
    for (const [t0, t1] of [
      [500.3, 1500.7],
      [800, 1300],
      [1210.2, 1311.9],
      [2000, 3000],
      [1000, 3500],
    ]) {
      const R = integrateGyro(s, t0, t1);
      const truth = mul(tr(att(t1)), att(t0)); // R_{d(t1)←d(t0)}
      const e = angleBetween(R, truth);
      worst = Math.max(worst, e);
      if (t1 - t0 <= 150) worstShort = Math.max(worstShort, e);
    }
    expect(worst).toBeLessThan(3e-5);
    expect(worstShort).toBeLessThan(1e-5);
  });

  it('higher-order step beats per-sample Euler integration', () => {
    const times: number[] = [];
    for (let i = 0; i < 120; i++) times.push(i * (1000 / 30)); // 30 Hz to make errors visible
    const s = samplesAt(times);
    const t0 = 100;
    const t1 = 3000;
    const truth = mul(tr(att(t1)), att(t0));
    const e = angleBetween(integrateGyro(s, t0, t1), truth);
    // naive: R ← R·exp(ω_i dt) with the left sample's rate
    let Rn: Float64Array = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    let ta = t0;
    for (const x of s) {
      if (x.t <= t0) continue;
      const tb = Math.min(x.t, t1);
      const prev = s[s.indexOf(x) - 1];
      const dt = (tb - ta) / 1000;
      Rn = mul(Rn, rotvecToMat3([prev.gyro[0] * dt, prev.gyro[1] * dt, prev.gyro[2] * dt]));
      ta = tb;
      if (tb >= t1) break;
    }
    const en = angleBetween(tr(Rn), truth);
    expect(e).toBeLessThan(en / 50);
  });

  it('reversed interval gives the inverse; no samples gives identity', () => {
    const s = samplesAt([0, 16, 33, 50, 66, 83, 100]);
    const R = integrateGyro(s, 10, 90);
    const Ri = integrateGyro(s, 90, 10);
    expect(angleBetween(mul(R, Ri), [1, 0, 0, 0, 1, 0, 0, 0, 1])).toBeLessThan(1e-12);
    expect(Array.from(integrateGyro([], 0, 100))).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    expect(Array.from(integrateGyro(s, 50, 50))).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });

  it('angularSpeedAt interpolates |ω|', () => {
    const s: ImuSample[] = [
      { t: 0, gyro: [1, 0, 0], accel: null, accelG: null },
      { t: 10, gyro: [0, 3, 0], accel: null, accelG: null },
    ];
    expect(angularSpeedAt(s, 5)).toBeCloseTo(2, 12);
    expect(angularSpeedAt(s, -50)).toBeCloseTo(1, 12);
    expect(angularSpeedAt(s, 500)).toBeNaN();
  });
});

describe('ImuRingBuffer', () => {
  const rec = new Float64Array(RAW_STRIDE);
  it('re-times a burst after a main-thread stall', () => {
    const b = new ImuRingBuffer(100);
    b.setNominal(16.667);
    const arrivals: number[] = [];
    for (let i = 0; i < 10; i++) arrivals.push(1000 + i * 16.667);
    // stall of 100 ms: 6 queued events dispatched back-to-back
    const T = arrivals[9] + 116;
    for (let k = 0; k < 6; k++) arrivals.push(T + 0.2 * k);
    for (let i = 0; i < 5; i++) arrivals.push(T + 1 + 16.667 * (i + 1));
    for (const a of arrivals) b.push(a, rec);
    expect(b.stats.bursts).toBeGreaterThanOrEqual(5);
    const times = Array.from({ length: b.size }, (_, i) => b.timeAt(b.first + i));
    for (let i = 1; i < times.length; i++) expect(times[i]).toBeGreaterThan(times[i - 1]);
    // pre-stall samples untouched
    for (let i = 0; i < 10; i++) expect(times[i]).toBeCloseTo(arrivals[i], 9);
    // burst spread at ≈ nominal ending at its last arrival
    for (let k = 11; k <= 15; k++) expect(times[k] - times[k - 1]).toBeGreaterThan(10);
    expect(times[15]).toBeCloseTo(T + 1, 9);
    expect(times[10]).toBeLessThan(T - 60);
  });
  it('wraps around and binary-searches by time', () => {
    const b = new ImuRingBuffer(16);
    for (let i = 0; i < 40; i++) b.push(i * 10, rec);
    expect(b.size).toBe(16);
    expect(b.first).toBe(24);
    expect(b.timeAt(b.first)).toBe(240);
    expect(b.lowerBound(255)).toBe(26);
    expect(b.lowerBound(0)).toBe(24);
    expect(b.lowerBound(1e9)).toBe(40);
  });
  it('estimates the nominal interval from arrivals and clamps non-monotonic stamps', () => {
    const b = new ImuRingBuffer(200);
    for (let i = 0; i < 100; i++) b.push(i * 10 + 0.5 * noise(i, 2), rec);
    expect(b.nominalMs).toBeGreaterThan(9);
    expect(b.nominalMs).toBeLessThan(11);
    b.push(500, rec);
    expect(b.stats.nonMonotonic).toBe(1);
  });
});

function ev(t: number, rr: number[], accG: number[] | null, acc: number[] | null = [0, 0, 0]): RawMotionEvent {
  return {
    timeStamp: t,
    interval: 1 / 60,
    rotationRate: { alpha: rr[0], beta: rr[1], gamma: rr[2] },
    acceleration: acc ? { x: acc[0], y: acc[1], z: acc[2] } : null,
    accelerationIncludingGravity: accG ? { x: accG[0], y: accG[1], z: accG[2] } : null,
  };
}

describe('MotionSensor (fed directly)', () => {
  function feed(sensor: MotionSensor, n: number, t0: number, withOri: boolean, beta = 80, gamma = 5): number[] {
    const up = deviceUpFromOrientation(beta, gamma);
    const times: number[] = [];
    for (let i = 0; i < n; i++) {
      const t = t0 + i * (1000 / 60);
      const w = omega(t);
      const accG = [-(G * up[0]) + 0.01 * noise(i, 1), -(G * up[1]) + 0.01 * noise(i, 2), -(G * up[2])]; // iOS sign
      // slow rotation for the first part keeps the stillness gate open
      const rr = i < 80 ? [0.2 * noise(i, 3), 0.2 * noise(i, 4), 0.2 * noise(i, 5)] : [w[0] * RAD, w[1] * RAD, w[2] * RAD];
      if (sensor.handleMotion(ev(t, rr, accG))) times.push(t);
      if (withOri) sensor.handleOrientation({ timeStamp: t + 0.3, alpha: 10, beta, gamma });
    }
    return times;
  }

  it('buffers, measures the rate, re-signs buffered samples once the sign is detected', () => {
    const m = new MotionSensor({ target: null, accelSignPrior: 1, screenAngle: () => 0 });
    expect(m.rateHz).toBe(0);
    const times = feed(m, 300, 1000, true);
    expect(times.length).toBe(299); // first replayed event dropped
    expect(m.rateHz).toBeGreaterThan(58);
    expect(m.rateHz).toBeLessThan(62);
    expect(m.calibration.accelSign).toBe(-1);
    // even the samples recorded before the decision now have the spec sign
    const early = m.samplesBetween(1000, 1100);
    expect(early.length).toBeGreaterThan(4);
    const up = deviceUpFromOrientation(80, 5);
    for (const s of early) {
      expect(s.accelG![1]).toBeGreaterThan(0.9 * G * up[1]);
      expect(s.gyro.length).toBe(3);
    }
  });

  it('samplesBetween (with/without padding), drain, angularSpeed, rotationBetween, gravityAt', () => {
    const m = new MotionSensor({ target: null, accelSignPrior: -1, screenAngle: () => 0 });
    feed(m, 200, 0, true);
    const all = m.samplesBetween(-1e9, 1e9);
    expect(all.length).toBe(m.sampleCount);
    const s = m.samplesBetween(2000, 2100);
    for (const x of s) {
      expect(x.t).toBeGreaterThanOrEqual(2000);
      expect(x.t).toBeLessThanOrEqual(2100);
    }
    const p = m.samplesBetween(2000, 2100, { pad: true });
    expect(p.length).toBe(s.length + 2);
    expect(p[0].t).toBeLessThan(2000);
    expect(p[p.length - 1].t).toBeGreaterThan(2100);

    const d1 = m.drain();
    expect(d1.length).toBe(m.sampleCount);
    expect(m.drain().length).toBe(0);
    feed(m, 10, 5000, false);
    const d2 = m.drain();
    expect(d2.length).toBe(10); // no reset() in between ⇒ nothing dropped
    expect(d2[0].t).toBeGreaterThanOrEqual(5000);

    // angular speed near t matches the analytic rate (gyro part of the trajectory)
    const t = 2500;
    const w = omega(t);
    expect(m.angularSpeed(t)).toBeCloseTo(Math.hypot(w[0], w[1], w[2]), 1);
    expect(m.angularSpeed(1e7)).toBeNaN();

    const R = m.rotationBetween(2000, 3000);
    expect(angleBetween(R, mul(tr(att(3000)), att(2000)))).toBeLessThan(2e-3);

    const g = m.gravityAt(1000)!;
    const up = deviceUpFromOrientation(80, 5);
    for (let k = 0; k < 3; k++) expect(g[k]).toBeCloseTo(-up[k], 9);
    // without orientation near t: from the specific force (spec sign ⇒ down = −accelG)
    const g2 = m.gravityAt(5100)!;
    for (let k = 0; k < 3; k++) expect(g2[k]).toBeCloseTo(-up[k], 2);
  });

  it('isSupported/needsPermission/requestPermission outside a browser', async () => {
    expect(MotionSensor.isSupported()).toBe(false);
    expect(MotionSensor.needsPermission()).toBe(false);
    expect(await MotionSensor.requestPermission()).toBe('unsupported');
  });

  it('requestPermission calls the iOS API synchronously and also asks DeviceOrientationEvent', async () => {
    const g = globalThis as Record<string, unknown>;
    const calls: string[] = [];
    g.DeviceMotionEvent = { requestPermission: () => (calls.push('motion'), Promise.resolve('granted')) };
    g.DeviceOrientationEvent = { requestPermission: () => (calls.push('orientation'), Promise.resolve('granted')) };
    try {
      const p = MotionSensor.requestPermission();
      expect(calls).toEqual(['motion']); // synchronous, before any await
      expect(await p).toBe('granted');
      expect(calls).toEqual(['motion', 'orientation']);
      expect(MotionSensor.needsPermission()).toBe(true);
      g.DeviceMotionEvent = { requestPermission: () => Promise.reject(Object.assign(new Error('gesture'), { name: 'NotAllowedError' })) };
      expect(await MotionSensor.requestPermission()).toBe('denied');
      expect(MotionSensor.lastPermissionError).toBe('NotAllowedError');
      g.DeviceMotionEvent = { requestPermission: () => Promise.resolve('denied') };
      expect(await MotionSensor.requestPermission()).toBe('denied');
      // iOS prior for the accelerometer sign
      expect(new MotionSensor({ target: null }).calibration.accelSign).toBe(-1);
    } finally {
      delete g.DeviceMotionEvent;
      delete g.DeviceOrientationEvent;
    }
  });

  it('start/stop attach listeners on the given target', () => {
    const target = new EventTarget();
    const m = new MotionSensor({ target, screenAngle: () => 0 });
    m.start();
    const mk = (t: number, a: number): Event => {
      const e = new Event('devicemotion');
      const fields = ev(t, [a, 0.5, 0.25], [0.1, 0.2, 9.7]) as unknown as Record<string, unknown>;
      for (const k of Object.keys(fields)) Object.defineProperty(e, k, { value: fields[k] });
      return e;
    };
    for (let i = 0; i < 5; i++) target.dispatchEvent(mk(i * 16, 1 + i * 0.123));
    expect(m.sampleCount).toBe(4);
    m.stop();
    target.dispatchEvent(mk(100, 9.87));
    expect(m.sampleCount).toBe(4);
  });
});
