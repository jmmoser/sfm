/**
 * IMU simulation (ARCHITECTURE §2 sign conventions):
 *   gyro   = ω_dev                       (body rate, dR_wd/dt = R_wd [ω_dev]×), rad/s
 *   accel  = R_wdᵀ a_world               (linear acceleration, gravity removed), m/s²
 *   accelG = R_wdᵀ (a_world − g_world)   (specific force, W3C sign), g_world = (0, 0, −9.81)
 * A device lying flat face-up at rest reports accelG ≈ (0, 0, +9.81).
 *
 * Noise: white Gaussian per sample (σ in the units above) plus constant biases. The
 * SAME accelerometer noise/bias realisation is applied to `accel` and `accelG`
 * (on iOS both come from one sensor; `acceleration` is derived by subtracting the
 * fused gravity estimate).
 *
 * Rate: iOS Safari delivers DeviceMotion at a fixed ~60 Hz (ARCHITECTURE §2.2), so that
 * is the default. `timeJitterMs` optionally perturbs the REPORTED timestamps (values stay
 * those of the regular true sample times), like event delivery jitter on a device.
 */
import type { ImuSample } from '../../core/types';
import { SynthRng } from './rng';
import type { Trajectory, TrajectoryState } from './trajectory';

export const GRAVITY = 9.81;

export interface ImuConfig {
  /** Sample rate (Hz). Default 60 (iOS DeviceMotion). */
  rateHz: number;
  /**
   * σ (ms) of Gaussian jitter added to the reported timestamps (clamped to ±2.5σ, kept
   * strictly increasing). Default 0 (timestamps exactly on the true clock).
   */
  timeJitterMs: number;
  /** Gyro white noise σ per sample (rad/s). Default 0.004. */
  gyroNoise: number;
  /** Accelerometer white noise σ per sample (m/s²). Default 0.04. */
  accelNoise: number;
  /** Constant gyro bias (rad/s, device frame). */
  gyroBias: [number, number, number];
  /** Constant accelerometer bias (m/s², device frame). */
  accelBias: [number, number, number];
  /** Noise seed. */
  seed: number;
}

export const DEFAULT_IMU_CONFIG: ImuConfig = {
  rateHz: 60,
  timeJitterMs: 0,
  gyroNoise: 0.004,
  accelNoise: 0.04,
  gyroBias: [0.003, -0.002, 0.0015],
  accelBias: [0.03, -0.02, 0.04],
  seed: 7,
};

/** Noise-free IMU reading for a trajectory state. */
export function idealImu(s: TrajectoryState, tMs: number): ImuSample {
  const R = s.R_wd,
    a = s.acceleration;
  // R_wdᵀ v
  const ax = R[0] * a[0] + R[3] * a[1] + R[6] * a[2];
  const ay = R[1] * a[0] + R[4] * a[1] + R[7] * a[2];
  const az = R[2] * a[0] + R[5] * a[1] + R[8] * a[2];
  // a − g = a + (0, 0, 9.81)
  const gx = ax + R[6] * GRAVITY,
    gy = ay + R[7] * GRAVITY,
    gz = az + R[8] * GRAVITY;
  return {
    t: tMs,
    gyro: [s.omegaDev[0], s.omegaDev[1], s.omegaDev[2]],
    accel: [ax, ay, az],
    accelG: [gx, gy, gz],
  };
}

/**
 * Sample the IMU at `rateHz` over trajectory time [startS, endS] (inclusive of start).
 * Timestamps: t = t0Ms + 1000·τ (IMU clock = true clock).
 */
export function simulateImu(
  traj: Trajectory,
  startS: number,
  endS: number,
  t0Ms: number,
  config: Partial<ImuConfig> = {},
): ImuSample[] {
  const c = { ...DEFAULT_IMU_CONFIG, ...config };
  const rng = new SynthRng(c.seed).derive('imu');
  const jrng = new SynthRng(c.seed).derive('imu-time-jitter');
  const n = Math.floor((endS - startS) * c.rateHz + 1e-9) + 1;
  const out: ImuSample[] = new Array(n);
  const minGapMs = 250 / c.rateHz;
  let prevT = -Infinity;
  for (let k = 0; k < n; k++) {
    const tau = startS + k / c.rateHz;
    const s = traj.evaluate(tau);
    let tMs = t0Ms + tau * 1000;
    if (c.timeJitterMs > 0) {
      const j = Math.max(-2.5, Math.min(2.5, jrng.gaussian())) * c.timeJitterMs;
      tMs = Math.max(tMs + j, prevT + minGapMs);
    }
    prevT = tMs;
    const m = idealImu(s, tMs);
    const g = m.gyro,
      a = m.accel!,
      ag = m.accelG!;
    for (let i = 0; i < 3; i++) {
      g[i] += c.gyroBias[i] + c.gyroNoise * rng.gaussian();
      const na = c.accelBias[i] + c.accelNoise * rng.gaussian();
      a[i] += na;
      ag[i] += na;
    }
    out[k] = m;
  }
  return out;
}
