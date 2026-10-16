/**
 * TEST-ONLY helpers for the sfm "extras" (inertial / geo / world) tests: continuous
 * synthetic trajectories (hand-held orbit, walk), IMU simulation, keyframes in an arbitrary
 * similarity frame. Not exported from the module index.
 *
 * World frame: Z-up, metres. Device frame / R_cam_device per ARCHITECTURE §2.
 */
import type { GeoFix, ImuSample, Mat3 } from '../core/types';
import { mat3Mul, mat3MulVec, mat3Transpose, mat3TransposeMul } from '../math/mat3';
import { Rng } from '../math/random';
import { mat3ToRotvec, rotvecToMat3 } from '../math/rotation';
import { EnuFrame } from './geo';
import type { InertialKeyframe } from './inertial';

export interface SimTrajectory {
  /** Duration (s). */
  duration: number;
  /** Camera centre (m, world Z-up) at time t (s). */
  pos(t: number, out: Float64Array): void;
  /** Camera-to-world rotation at time t (s). */
  Rwc(t: number, out: Float64Array): void;
}

/** R_cam_device(θ) = diag(1, −1, −1) · Rz(θ). */
export function cameraFromDevice(thetaDeg: number): Mat3 {
  const th = (thetaDeg * Math.PI) / 180;
  const c = Math.round(Math.cos(th) * 1e12) / 1e12,
    s = Math.round(Math.sin(th) * 1e12) / 1e12;
  return Float64Array.of(c, -s, 0, -s, -c, 0, 0, 0, -1);
}

/** Camera-to-world rotation looking along `f` with world +Z as up hint (columns right, down, forward). */
export function lookAt(fx: number, fy: number, fz: number, out: Float64Array): void {
  const fn = Math.hypot(fx, fy, fz);
  fx /= fn;
  fy /= fn;
  fz /= fn;
  // right = f × up(0,0,1)
  let rx = fy,
    ry = -fx;
  const rn = Math.hypot(rx, ry) || 1;
  rx /= rn;
  ry /= rn;
  const rz = 0;
  // down = f × right
  const dx = fy * rz - fz * ry,
    dy = fz * rx - fx * rz,
    dz = fx * ry - fy * rx;
  out[0] = rx;
  out[1] = dx;
  out[2] = fx;
  out[3] = ry;
  out[4] = dy;
  out[5] = fy;
  out[6] = rz;
  out[7] = dz;
  out[8] = fz;
}

interface Sines {
  a: Float64Array;
  f: Float64Array;
  p: Float64Array;
}

function sines(rng: Rng, n: number, amp: [number, number], freq: [number, number]): Sines {
  const s = { a: new Float64Array(n), f: new Float64Array(n), p: new Float64Array(n) };
  for (let i = 0; i < n; i++) {
    s.a[i] = rng.uniform(amp[0], amp[1]);
    s.f[i] = rng.uniform(freq[0], freq[1]);
    s.p[i] = rng.uniform(0, 2 * Math.PI);
  }
  return s;
}

function evalSines(s: Sines, t: number): number {
  let v = 0;
  for (let i = 0; i < s.a.length; i++) v += s.a[i] * Math.sin(2 * Math.PI * s.f[i] * t + s.p[i]);
  return v;
}

/**
 * Hand-held orbit around the origin: mean radius 0.4 m at `orbitHz` (0.2), with the
 * irregularities of a real hand-held orbit — angular speed varying ±15 %, radius ±3 cm,
 * height ±3 cm — plus position jitter (3 sinusoids per axis, `posJitterMm` amplitude,
 * 0.4–2.5 Hz) and rotation jitter.
 */
export function orbitTrajectory(
  rng: Rng,
  opts: { duration?: number; radius?: number; orbitHz?: number; posJitterMm?: [number, number]; rotJitterDeg?: [number, number]; regular?: boolean } = {},
): SimTrajectory {
  const duration = opts.duration ?? 20;
  const r0 = opts.radius ?? 0.4;
  const fo = opts.orbitHz ?? 0.2;
  const pj = opts.posJitterMm ?? [3, 6];
  const rj = opts.rotJitterDeg ?? [0.2, 0.6];
  const irr = opts.regular ? 0 : 1;
  const th0 = rng.uniform(0, 2 * Math.PI);
  const fs = rng.uniform(0.3, 0.7),
    ps = rng.uniform(0, 2 * Math.PI);
  const fr = rng.uniform(0.2, 0.5),
    pr = rng.uniform(0, 2 * Math.PI);
  const fh = rng.uniform(0.1, 0.3),
    ph = rng.uniform(0, 2 * Math.PI);
  const jx = sines(rng, 3, [pj[0] / 1000, pj[1] / 1000], [0.4, 2.5]);
  const jy = sines(rng, 3, [pj[0] / 1000, pj[1] / 1000], [0.4, 2.5]);
  const jz = sines(rng, 3, [pj[0] / 1000, pj[1] / 1000], [0.4, 2.5]);
  const D = Math.PI / 180;
  const rx = sines(rng, 3, [rj[0] * D, rj[1] * D], [0.5, 3]);
  const ry = sines(rng, 3, [rj[0] * D, rj[1] * D], [0.5, 3]);
  const rz = sines(rng, 3, [rj[0] * D, rj[1] * D], [0.5, 3]);
  const tw = sines(rng, 2, [0.01, 0.02], [0.05, 0.15]);
  const L = new Float64Array(9),
    J = new Float64Array(9),
    v = new Float64Array(3),
    p = new Float64Array(3);
  const w0 = 2 * Math.PI * fo;
  const traj: SimTrajectory = {
    duration,
    pos(t, out) {
      // θ' = w0 (1 + 0.15 sin(2π fs t + ps))
      const th = th0 + w0 * t - irr * ((0.15 * w0) / (2 * Math.PI * fs)) * (Math.cos(2 * Math.PI * fs * t + ps) - Math.cos(ps));
      const r = r0 + irr * 0.03 * Math.sin(2 * Math.PI * fr * t + pr);
      out[0] = r * Math.cos(th) + evalSines(jx, t);
      out[1] = r * Math.sin(th) + evalSines(jy, t);
      out[2] = 0.25 + 0.03 * Math.sin(2 * Math.PI * 0.07 * t) + irr * 0.03 * Math.sin(2 * Math.PI * fh * t + ph) + evalSines(jz, t);
    },
    Rwc(t, out) {
      traj.pos(t, p);
      const tx = evalSines(tw, t),
        ty = evalSines(tw, t + 3.3);
      lookAt(tx - p[0], ty - p[1], -p[2], L);
      v[0] = evalSines(rx, t);
      v[1] = evalSines(ry, t);
      v[2] = evalSines(rz, t);
      rotvecToMat3(v, J);
      mat3Mul(L, J, out);
    },
  };
  return traj;
}

/** Walk: 1 s rest, 2 s ramp to 1.2 m/s, cruise, 2 s ramp down, 1 s rest; 40 m along a gentle arc; 2 Hz bob, 1 Hz sway. */
export function walkTrajectory(rng: Rng, opts: { distance?: number; speed?: number; lookLeftDeg?: number } = {}): SimTrajectory {
  const dist = opts.distance ?? 40;
  const v = opts.speed ?? 1.2;
  const ramp = 2;
  const t1 = 1,
    t2 = t1 + ramp;
  const sRamp = (v / 2) * ramp;
  const Tc = (dist - 2 * sRamp) / v;
  const t3 = t2 + Tc,
    t4 = t3 + ramp;
  const duration = t4 + 1;
  const psi0 = rng.uniform(0, 2 * Math.PI);
  const kappa = 1 / 120;
  const D = Math.PI / 180;
  const look = (opts.lookLeftDeg ?? 30) * D;
  const rj = [sines(rng, 3, [0.1 * D, 0.3 * D], [0.5, 3]), sines(rng, 3, [0.1 * D, 0.3 * D], [0.5, 3]), sines(rng, 3, [0.1 * D, 0.3 * D], [0.5, 3])];
  const pj = [sines(rng, 2, [0.001, 0.003], [0.5, 2]), sines(rng, 2, [0.001, 0.003], [0.5, 2]), sines(rng, 2, [0.001, 0.003], [0.5, 2])];
  const sOf = (t: number): number => {
    if (t < t1) return 0;
    if (t < t2) {
      const u = t - t1;
      return (v / 2) * (u - (ramp / Math.PI) * Math.sin((Math.PI * u) / ramp));
    }
    if (t < t3) return sRamp + v * (t - t2);
    if (t < t4) {
      const u = t - t3;
      return sRamp + v * Tc + (v / 2) * (u + (ramp / Math.PI) * Math.sin((Math.PI * u) / ramp));
    }
    return dist;
  };
  const speedFactor = (t: number): number => {
    if (t < t1 || t >= t4) return 0;
    if (t < t2) return 0.5 * (1 - Math.cos((Math.PI * (t - t1)) / ramp));
    if (t < t3) return 1;
    return 0.5 * (1 + Math.cos((Math.PI * (t - t3)) / ramp));
  };
  const L = new Float64Array(9),
    J = new Float64Array(9),
    w = new Float64Array(3);
  const traj: SimTrajectory = {
    duration,
    pos(t, out) {
      const A = speedFactor(t);
      const s0 = sOf(t);
      const s = s0 + 0.008 * A * Math.sin((2 * Math.PI * s0) / 0.6 + 0.5);
      const psi = psi0 + kappa * s;
      let x = (Math.sin(psi) - Math.sin(psi0)) / kappa;
      let y = -(Math.cos(psi) - Math.cos(psi0)) / kappa;
      const sway = 0.02 * A * Math.sin((Math.PI * s0) / 0.6);
      x += -Math.sin(psi) * sway;
      y += Math.cos(psi) * sway;
      out[0] = x + evalSines(pj[0], t);
      out[1] = y + evalSines(pj[1], t);
      out[2] = 1.5 + 0.025 * A * Math.sin((2 * Math.PI * s0) / 0.6) + evalSines(pj[2], t);
    },
    Rwc(t, out) {
      const A = speedFactor(t);
      const s0 = sOf(t);
      const psi = psi0 + kappa * s0 + look + 1.5 * D * A * Math.sin((Math.PI * s0) / 0.6);
      const pitch = -8 * D + 1 * D * A * Math.sin((2 * Math.PI * s0) / 0.6 + 1);
      lookAt(Math.cos(pitch) * Math.cos(psi), Math.cos(pitch) * Math.sin(psi), Math.sin(pitch), L);
      w[0] = evalSines(rj[0], t);
      w[1] = evalSines(rj[1], t);
      w[2] = 0.8 * D * A * Math.sin((Math.PI * s0) / 0.6 + 0.3) + evalSines(rj[2], t);
      rotvecToMat3(w, J);
      mat3Mul(L, J, out);
    },
  };
  return traj;
}

/** Smooth, low-acceleration pan (for "insufficient excitation"). */
export function slowPanTrajectory(duration = 15): SimTrajectory {
  const L = new Float64Array(9);
  return {
    duration,
    pos(t, out) {
      out[0] = 0.02 * t;
      out[1] = 0;
      out[2] = 1.5;
    },
    Rwc(t, out) {
      const psi = 0.05 * t;
      lookAt(Math.cos(psi), Math.sin(psi), -0.1, L);
      out.set(L);
    },
  };
}

/** World acceleration by a 5-point stencil (h = 5 ms). */
export function trajAccel(traj: SimTrajectory, t: number, out: Float64Array): void {
  const h = 0.005;
  const a = new Float64Array(3),
    b = new Float64Array(3),
    c = new Float64Array(3),
    d = new Float64Array(3),
    e = new Float64Array(3);
  traj.pos(t - 2 * h, a);
  traj.pos(t - h, b);
  traj.pos(t, c);
  traj.pos(t + h, d);
  traj.pos(t + 2 * h, e);
  for (let i = 0; i < 3; i++) out[i] = (-a[i] + 16 * b[i] - 30 * c[i] + 16 * d[i] - e[i]) / (12 * h * h);
}

/** Device-frame angular velocity: log(R_wd(t−h)ᵀ R_wd(t+h)) / 2h. */
export function trajOmega(traj: SimTrajectory, Rcd: Mat3, t: number, out: Float64Array): void {
  const h = 1e-4;
  const A = new Float64Array(9),
    B = new Float64Array(9);
  traj.Rwc(t - h, A);
  traj.Rwc(t + h, B);
  const Ra = mat3Mul(A, Rcd),
    Rb = mat3Mul(B, Rcd);
  const rel = mat3TransposeMul(Ra, Rb);
  mat3ToRotvec(rel, out);
  for (let i = 0; i < 3; i++) out[i] /= 2 * h;
}

export interface ImuSimOptions {
  rateHz: number;
  t0Ms: number;
  jitterMs: number;
  gyroNoise: number;
  accelNoise: number;
  gyroBias: [number, number, number];
  accelBias: [number, number, number];
  withAccel: boolean;
  withAccelG: boolean;
}

export const DEFAULT_IMU_SIM: ImuSimOptions = {
  rateHz: 100,
  t0Ms: 5000,
  jitterMs: 0.5,
  gyroNoise: 0.005,
  accelNoise: 0.05,
  gyroBias: [0.008, -0.006, 0.004],
  accelBias: [0.03, -0.02, 0.025],
  withAccel: true,
  withAccelG: true,
};

/** IMU samples over the trajectory (ms timestamps t0Ms + 1000 t). */
export function simulateImu(traj: SimTrajectory, Rcd: Mat3, rng: Rng, opts: Partial<ImuSimOptions> = {}): ImuSample[] {
  const o = { ...DEFAULT_IMU_SIM, ...opts };
  const out: ImuSample[] = [];
  const n = Math.floor(traj.duration * o.rateHz);
  const a = new Float64Array(3),
    w = new Float64Array(3),
    Rc = new Float64Array(9);
  for (let k = 0; k <= n; k++) {
    const t = Math.min(traj.duration, Math.max(0, k / o.rateHz + (rng.next() - 0.5) * 2 * (o.jitterMs / 1000)));
    trajAccel(traj, t, a);
    trajOmega(traj, Rcd, t, w);
    traj.Rwc(t, Rc);
    const Rwd = mat3Mul(Rc, Rcd);
    const RwdT = mat3Transpose(Rwd);
    const lin = mat3MulVec(RwdT, a);
    const spec = mat3MulVec(RwdT, Float64Array.of(a[0], a[1], a[2] + 9.81));
    const na = [o.accelNoise * rng.gaussian(), o.accelNoise * rng.gaussian(), o.accelNoise * rng.gaussian()];
    out.push({
      t: o.t0Ms + 1000 * t,
      gyro: [
        w[0] + o.gyroBias[0] + o.gyroNoise * rng.gaussian(),
        w[1] + o.gyroBias[1] + o.gyroNoise * rng.gaussian(),
        w[2] + o.gyroBias[2] + o.gyroNoise * rng.gaussian(),
      ],
      accel: o.withAccel ? [lin[0] + o.accelBias[0] + na[0], lin[1] + o.accelBias[1] + na[1], lin[2] + o.accelBias[2] + na[2]] : null,
      accelG: o.withAccelG ? [spec[0] + o.accelBias[0] + na[0], spec[1] + o.accelBias[1] + na[1], spec[2] + o.accelBias[2] + na[2]] : null,
    });
  }
  return out;
}

export function randomRotation(rng: Rng, maxAngle = Math.PI): Mat3 {
  const ax = [rng.gaussian(), rng.gaussian(), rng.gaussian()];
  const n = Math.hypot(ax[0], ax[1], ax[2]);
  const ang = rng.uniform(0, maxAngle);
  return rotvecToMat3([(ax[0] / n) * ang, (ax[1] / n) * ang, (ax[2] / n) * ang]);
}

export interface KeyframeSimOptions {
  minIntervalS: number;
  maxIntervalS: number;
  /** True offset: t_imu = t_cam + offsetMs. */
  offsetMs: number;
  t0Ms: number;
  /** Similarity of the reconstruction frame: X_rec = s R X + t. */
  s: number;
  R: Mat3 | null;
  t: [number, number, number];
  /** Position noise (m, metric) and rotation noise (rad). */
  posNoise: number;
  rotNoise: number;
  marginS: number;
}

export interface SimKeyframes {
  keyframes: (InertialKeyframe & { id: number; tTrue: number })[];
  s: number;
  R: Mat3;
  t: Float64Array;
  /** Gravity DOWN in the reconstruction frame. */
  downRec: Float64Array;
  /** True metric camera centres per keyframe (3n). */
  centresTrue: Float64Array;
}

export function simulateKeyframes(traj: SimTrajectory, Rcd: Mat3, rng: Rng, opts: Partial<KeyframeSimOptions> = {}): SimKeyframes {
  const o: KeyframeSimOptions = {
    minIntervalS: 0.25,
    maxIntervalS: 0.5,
    offsetMs: -37,
    t0Ms: 5000,
    s: 2.7,
    R: null,
    t: [3, -1, 7],
    posNoise: 0.0005,
    rotNoise: 0.0005,
    marginS: 0.4,
    ...opts,
  };
  const R0 = o.R ?? randomRotation(rng);
  const t0 = Float64Array.from(o.t);
  const kfs: SimKeyframes['keyframes'] = [];
  const centres: number[] = [];
  const p = new Float64Array(3),
    Rc = new Float64Array(9),
    nz = new Float64Array(3),
    Nm = new Float64Array(9);
  let t = o.marginS;
  let id = 0;
  while (t < traj.duration - o.marginS) {
    traj.pos(t, p);
    traj.Rwc(t, Rc);
    centres.push(p[0], p[1], p[2]);
    nz[0] = o.rotNoise * rng.gaussian();
    nz[1] = o.rotNoise * rng.gaussian();
    nz[2] = o.rotNoise * rng.gaussian();
    rotvecToMat3(nz, Nm);
    const Rrec = mat3Mul(mat3Mul(R0, Rc), Nm);
    const Cm = mat3MulVec(R0, p);
    const C = new Float64Array(3);
    for (let i = 0; i < 3; i++) C[i] = o.s * Cm[i] + t0[i] + o.s * o.posNoise * rng.gaussian();
    kfs.push({ id: id++, t: o.t0Ms + 1000 * t - o.offsetMs, tTrue: o.t0Ms + 1000 * t, R_wc: Rrec, C, R_cam_device: Rcd });
    t += rng.uniform(o.minIntervalS, o.maxIntervalS);
  }
  return { keyframes: kfs, s: o.s, R: R0, t: t0, downRec: mat3MulVec(R0, [0, 0, -1]), centresTrue: Float64Array.from(centres) };
}

/** GPS fixes at 1 Hz along the trajectory with white horizontal noise σ (m), vertical 1.5σ. */
export function simulateGps(
  traj: SimTrajectory,
  rng: Rng,
  opts: { origin?: [number, number, number]; sigma?: number; t0Ms?: number; rateHz?: number; withAlt?: boolean; correlatedSigma?: number; tauS?: number } = {},
): { fixes: GeoFix[]; frame: EnuFrame } {
  const origin = opts.origin ?? [47.3769, 8.5417, 410];
  const sigma = opts.sigma ?? 3;
  const frame = new EnuFrame(origin[0], origin[1], origin[2]);
  const rate = opts.rateHz ?? 1;
  const fixes: GeoFix[] = [];
  const p = new Float64Array(3);
  const cs = opts.correlatedSigma ?? 0;
  const tau = opts.tauS ?? 10;
  const phi = Math.exp(-1 / (rate * tau));
  const cx = [cs * rng.gaussian(), cs * rng.gaussian(), cs * rng.gaussian()];
  for (let t = 0; t <= traj.duration; t += 1 / rate) {
    traj.pos(t, p);
    for (let i = 0; i < 3; i++) cx[i] = phi * cx[i] + Math.sqrt(1 - phi * phi) * cs * rng.gaussian();
    const e = p[0] + sigma * rng.gaussian() + cx[0];
    const n = p[1] + sigma * rng.gaussian() + cx[1];
    const u = p[2] + 1.5 * sigma * rng.gaussian() + cx[2];
    const g = frame.toGeodetic(e, n, u);
    fixes.push({
      t: (opts.t0Ms ?? 5000) + 1000 * t,
      lat: g[0],
      lon: g[1],
      alt: opts.withAlt === false ? null : g[2],
      accuracy: sigma,
      altitudeAccuracy: opts.withAlt === false ? null : 1.5 * sigma,
    });
  }
  return { fixes, frame };
}

export function angleDeg(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const c = Math.hypot(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
  return (Math.atan2(c, d) * 180) / Math.PI;
}
