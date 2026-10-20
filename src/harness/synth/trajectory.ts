/**
 * Smooth device trajectories with EXACT analytic first and second derivatives of
 * position and orientation (forward-mode jets, see jet.ts). Everything is a finite sum
 * of sinusoids composed with atan2/sqrt away from their singularities, hence C∞.
 *
 * Frames (ARCHITECTURE §2): world Z-up ENU, metres; R_wd = device-to-world rotation;
 * dR_wd/dt = R_wd [ω_dev]×; camera pose R = R_cam_device(θ) · R_wdᵀ, centre = position
 * (camera and IMU are co-located — no lever arm).
 */
import type { Mat3, Pose, Vec3 } from '../../core/types';
import {
  type Jet,
  type JetMat3,
  type JetVec3,
  jadd,
  jaddc,
  jatan2,
  jconst,
  jcos,
  jdiv,
  jgaussian,
  jmatConst,
  jmatMul,
  jmatSplit,
  jmul,
  jneg,
  jrotY,
  jrotZ,
  jscale,
  jsin,
  jsinusoid,
  jsqrt,
  jsub,
  jsum,
  jvar,
  jvecSplit,
} from './jet';
import { m3Add, m3Mul, m3TMul, vee } from './mat';
import { type ScreenAngle, poseFromDevice, rCamDevice } from './conventions';

export interface TrajectoryState {
  /** Seconds since the trajectory start. */
  t: number;
  /** Device / camera centre in the world frame (m). */
  position: Vec3;
  /** dp/dt (m/s), world frame. */
  velocity: Vec3;
  /** d²p/dt² (m/s²), world frame (no gravity). */
  acceleration: Vec3;
  /** Device-to-world rotation. */
  R_wd: Mat3;
  /** dR_wd/dt */
  dR_wd: Mat3;
  /** d²R_wd/dt² */
  ddR_wd: Mat3;
  /** Body angular velocity (device frame, rad/s): dR_wd/dt = R_wd [ω]×. */
  omegaDev: Vec3;
  /** Body angular acceleration dω_dev/dt (device frame, rad/s²). */
  omegaDotDev: Vec3;
}

export interface Trajectory {
  readonly kind: string;
  /** Nominal duration (s). The trajectory is defined (smoothly) for all t. */
  readonly duration: number;
  /** UI rotation θ of the frames (ARCHITECTURE §2). */
  readonly screenAngle: ScreenAngle;
  evaluate(t: number): TrajectoryState;
}

/** Jet-valued trajectory definition: position and device attitude as functions of t. */
export type JetTrajectoryFn = (t: Jet) => { p: JetVec3; R_wd: JetMat3 };

/** Build a Trajectory from a jet function (exact derivatives). */
export function createJetTrajectory(
  kind: string,
  duration: number,
  screenAngle: ScreenAngle,
  fn: JetTrajectoryFn,
): Trajectory {
  return {
    kind,
    duration,
    screenAngle,
    evaluate(t: number): TrajectoryState {
      const { p, R_wd } = fn(jvar(t));
      const pv = jvecSplit(p);
      const R = jmatSplit(R_wd);
      // Ω = Rᵀ Ṙ (skew), Ω̇ = Ṙᵀ Ṙ + Rᵀ R̈.
      const Omega = m3TMul(R.v, R.d);
      const OmegaDot = m3Add(m3TMul(R.d, R.d), m3TMul(R.v, R.dd));
      return {
        t,
        position: pv.v,
        velocity: pv.d,
        acceleration: pv.dd,
        R_wd: R.v,
        dR_wd: R.d,
        ddR_wd: R.dd,
        omegaDev: vee(Omega),
        omegaDotDev: vee(OmegaDot),
      };
    },
  };
}

/** Camera pose (world-to-camera) at time t. */
export function trajectoryPose(traj: Trajectory, t: number): Pose {
  const s = traj.evaluate(t);
  return poseFromDevice(s.R_wd, s.position, traj.screenAngle);
}

/**
 * Camera-to-world rotation of a camera with the given yaw/pitch/roll:
 * R_wc = Rz(yaw) · Ry(−pitch) · B · Rz(roll), where B maps the camera axes of a level
 * camera looking along world +X (forward +X, right −Y, down −Z). yaw is measured
 * counter-clockwise from east (+X) towards north; pitch > 0 looks up; roll rotates
 * about the optical axis.
 */
export function cameraToWorldJet(yaw: Jet, pitch: Jet, roll: Jet): JetMat3 {
  const B = jmatConst([0, 0, 1, -1, 0, 0, 0, -1, 0]);
  return jmatMul(jmatMul(jmatMul(jrotZ(yaw), jrotY(jneg(pitch))), B), jrotZ(roll));
}

/** R_wd = R_wc · R_cam_device(θ)  (from R = R_cam_device R_wdᵀ, R_wc = Rᵀ). */
export function deviceAttitudeJet(R_wc: JetMat3, screenAngle: ScreenAngle): JetMat3 {
  return jmatMul(R_wc, jmatConst(rCamDevice(screenAngle)));
}

/** Σ amplitude·sin(2π f t + φ) */
function sines(t: Jet, terms: readonly (readonly [number, number, number])[]): Jet {
  return jsum(...terms.map(([a, f, ph]) => jsinusoid(t, a, f, ph)));
}

// ---------------------------------------------------------------------------
// Hand-held shake (optional, both trajectories): physiological tremor + fast pans
// ---------------------------------------------------------------------------

/**
 * A fast "glance" pan: Gaussian out-and-back bump added to the camera yaw / pitch,
 * offset(t) = A·exp(−(t − t)² / (2σ²)). Peak angular speed ≈ 0.6065·hypot(yawRad, pitchRad)/σ
 * (reached at t ± σ), on top of the trajectory's own motion.
 */
export interface PanEvent {
  /** Centre time (s). */
  t: number;
  /** Yaw amplitude (rad, + = counter-clockwise seen from above). */
  yawRad: number;
  /** Pitch amplitude (rad, + = up). */
  pitchRad: number;
  /** Gaussian σ (s). */
  sigmaS: number;
}

export interface ShakeOptions {
  /**
   * Rotational hand tremor (deg): amplitude of the strongest of three 8–12 Hz sinusoids per
   * axis (yaw, pitch, roll), slowly amplitude-modulated (±40 %, 3.3 s). 0 = off.
   */
  tremorDeg: number;
  /** Positional tremor (m): same structure, per world axis. 0 = off. */
  tremorM: number;
  /** Occasional fast pans. */
  pans: PanEvent[];
}

/** [relative amplitude, Hz, phase] of the 8–12 Hz tremor components per axis. */
const TREMOR_ROT: readonly (readonly [number, number, number])[][] = [
  [[1, 8.7, 0.3], [0.7, 10.3, 2.1], [0.45, 11.6, 4.4]], // yaw
  [[1, 9.1, 1.2], [0.6, 10.9, 3.3], [0.4, 8.2, 5.0]], // pitch
  [[0.8, 9.6, 0.7], [0.5, 11.2, 2.8], [0.3, 8.4, 4.1]], // roll
];
const TREMOR_POS: readonly (readonly [number, number, number])[][] = [
  [[1, 9.3, 0.9], [0.6, 11.4, 3.7], [0.4, 8.1, 1.9]],
  [[1, 8.6, 2.6], [0.6, 10.6, 0.2], [0.4, 11.9, 4.8]],
  [[0.8, 10.1, 5.3], [0.5, 8.9, 1.4], [0.3, 11.1, 3.0]],
];

function tremorTerms(t: Jet, amp: number, table: readonly (readonly [number, number, number])[][], modPhase: number): Jet[] {
  // Slow amplitude modulation: tremor comes and goes.
  const mod = jaddc(jsinusoid(t, 0.4, 1 / 3.3, modPhase), 1);
  return table.map((terms) => jmul(mod, sines(t, terms.map(([a, f, ph]) => [a * amp, f, ph] as const))));
}

/** Yaw / pitch / roll offsets (rad) of the shake model, or null when off. */
export function shakeAngles(t: Jet, s: Partial<ShakeOptions> | undefined | null): { yaw: Jet; pitch: Jet; roll: Jet } | null {
  if (!s) return null;
  const tremor = s.tremorDeg ?? 0;
  const pans = s.pans ?? [];
  if (tremor === 0 && pans.length === 0) return null;
  let yaw = jconst(0),
    pitch = jconst(0),
    roll = jconst(0);
  if (tremor !== 0) {
    const [y, p, r] = tremorTerms(t, (tremor * Math.PI) / 180, TREMOR_ROT, 0.5);
    yaw = jadd(yaw, y);
    pitch = jadd(pitch, p);
    roll = jadd(roll, r);
  }
  for (const e of pans) {
    if (e.yawRad !== 0) yaw = jadd(yaw, jgaussian(t, e.yawRad, e.t, e.sigmaS));
    if (e.pitchRad !== 0) pitch = jadd(pitch, jgaussian(t, e.pitchRad, e.t, e.sigmaS));
  }
  return { yaw, pitch, roll };
}

/** World-frame positional tremor offset (m), or null when off. */
export function shakeOffset(t: Jet, s: Partial<ShakeOptions> | undefined | null): JetVec3 | null {
  if (!s || !s.tremorM) return null;
  const [x, y, z] = tremorTerms(t, s.tremorM, TREMOR_POS, 2.2);
  return [x, y, z];
}

/** Peak analytic slope of a pan's angle offset (rad/s). */
export function panPeakRate(e: PanEvent): number {
  return (Math.exp(-0.5) * Math.hypot(e.yawRad, e.pitchRad)) / e.sigmaS;
}

// ---------------------------------------------------------------------------
// 'object': hand-held orbit around a ~25 cm object on a table
// ---------------------------------------------------------------------------

export interface ObjectTrajectoryOptions {
  /** Look-at target (object centre), world. Default (0, 0, 0.86). */
  target: [number, number, number];
  /** Mean orbit radius (m). Default 0.45. */
  radius: number;
  /** Revolutions over `duration`. Default 1.2. */
  revolutions: number;
  /** Seconds. Default 25. */
  duration: number;
  /** Mean camera height above the target (m). Default 0.2. */
  heightAbove: number;
  /** Slow height oscillation amplitude (m). Default 0.09. */
  heightAmplitude: number;
  /** Orbit start angle (rad, CCW from +X). Default −π/2 (camera south of the object). */
  startAngle: number;
  /** Scale of the hand-held jitter (positional tremor + rotational wobble). Default 1. */
  jitter: number;
  /** UI rotation of the frames. Default 0 (portrait). */
  screenAngle: ScreenAngle;
  /** Extra hand-held shake (tremor 8–12 Hz, fast pans). Default null (off). */
  shake: Partial<ShakeOptions> | null;
}

export const DEFAULT_OBJECT_TRAJECTORY: ObjectTrajectoryOptions = {
  target: [0, 0, 0.86],
  radius: 0.45,
  revolutions: 1.2,
  duration: 25,
  heightAbove: 0.2,
  heightAmplitude: 0.09,
  startAngle: -Math.PI / 2,
  jitter: 1,
  screenAngle: 0,
  shake: null,
};

export function createObjectTrajectory(opts: Partial<ObjectTrajectoryOptions> = {}): Trajectory {
  const o = { ...DEFAULT_OBJECT_TRAJECTORY, ...opts };
  const T = o.duration;
  const omega = (2 * Math.PI * o.revolutions) / T;
  const J = o.jitter;
  // Positional hand tremor (m): [amplitude, Hz, phase] per axis.
  const trem: (readonly [number, number, number])[][] = [
    [[0.002 * J, 1.3, 0.1], [0.001 * J, 2.9, 1.7], [0.0003 * J, 5.3, 4.0]],
    [[0.002 * J, 1.1, 2.3], [0.001 * J, 3.3, 0.6], [0.0003 * J, 4.7, 2.9]],
    [[0.0015 * J, 1.7, 5.1], [0.0008 * J, 2.3, 3.2], [0.0003 * J, 6.1, 1.2]],
  ];
  const fn: JetTrajectoryFn = (t) => {
    const phi = jsum(jconst(o.startAngle), jscale(t, omega), jsinusoid(t, 0.12, 1 / 9.3, 0.4));
    const r = jsum(jconst(o.radius), jsinusoid(t, 0.03, 1 / 7.1, 1.0), jsinusoid(t, 0.012, 1 / 3.3, 0.2));
    const h = jsum(
      jconst(o.heightAbove),
      jsinusoid(t, o.heightAmplitude, 1.5 / T, 0.3),
      jsinusoid(t, 0.015, 1 / 4.3, 2.0),
    );
    const p: JetVec3 = [
      jsum(jconst(o.target[0]), jmul(r, jcos(phi)), sines(t, trem[0])),
      jsum(jconst(o.target[1]), jmul(r, jsin(phi)), sines(t, trem[1])),
      jsum(jconst(o.target[2]), h, sines(t, trem[2])),
    ];
    const shakeP = shakeOffset(t, o.shake);
    // Positional shake moves the camera only (the aim point stays): added after the look-at.
    const pCam: JetVec3 = shakeP ? [jadd(p[0], shakeP[0]), jadd(p[1], shakeP[1]), jadd(p[2], shakeP[2])] : p;
    // Look-at with a slowly wandering aim point (the object is not always centred).
    const aim: JetVec3 = [
      jaddc(jsinusoid(t, 0.015, 1 / 6.1, 0), o.target[0]),
      jaddc(jsinusoid(t, 0.015, 1 / 7.7, 1.0), o.target[1]),
      jaddc(jsinusoid(t, 0.01, 1 / 5.3, 2.0), o.target[2]),
    ];
    const dx = jsub(aim[0], p[0]),
      dy = jsub(aim[1], p[1]),
      dz = jsub(aim[2], p[2]);
    const horiz = jsqrt(jadd(jmul(dx, dx), jmul(dy, dy)));
    let yaw = jsum(jatan2(dy, dx), sines(t, [[0.005 * J, 2.3, 0.4], [0.003 * J, 4.1, 2.2]]));
    let pitch = jsum(jatan2(dz, horiz), sines(t, [[0.004 * J, 3.1, 1.0], [0.002 * J, 1.7, 0]]));
    let roll = sines(t, [[0.035, 1 / 5.7, 0.3], [0.01 * J, 1.9, 1.7]]);
    const sh = shakeAngles(t, o.shake);
    if (sh) {
      yaw = jadd(yaw, sh.yaw);
      pitch = jadd(pitch, sh.pitch);
      roll = jadd(roll, sh.roll);
    }
    const R_wd = deviceAttitudeJet(cameraToWorldJet(yaw, pitch, roll), o.screenAngle);
    return { p: pCam, R_wd };
  };
  return createJetTrajectory('object', T, o.screenAngle, fn);
}

// ---------------------------------------------------------------------------
// 'yard': person walking a ~15 x 10 m loop, phone held at ~1.5 m looking outward
// ---------------------------------------------------------------------------

export interface YardTrajectoryOptions {
  /** Loop centre (x, y). Default (0, 0). */
  center: [number, number];
  /** Semi-axes of the (perturbed) elliptical loop (m). Default [7.5, 5]. */
  semiAxes: [number, number];
  /** Laps over `duration` (slightly > 1 ⇒ the walk passes its start: loop closure). Default 1.03. */
  laps: number;
  /** Seconds. Default 40 (≈ 1 m/s). */
  duration: number;
  /** Phone height above ground (m). Default 1.5. */
  height: number;
  /** Gait bob amplitude (m) at `stepHz`. Default 0.03. */
  bobAmplitude: number;
  /** Step frequency (Hz). Default 2. */
  stepHz: number;
  /** Lateral sway amplitude (m) at stepHz/2. Default 0.02. */
  swayAmplitude: number;
  /**
   * Constant sideways offset of the walk from the loop (m, along the outward loop normal; + = outward).
   * The scene's gravel path stays on the loop, so test sets of the same seeded scene can walk beside
   * it (e2e/README.md "Off-trajectory views"). Default 0.
   */
  lateralOffset: number;
  /** Slow panning amplitude around the outward direction (deg). Default 25. */
  panAmplitudeDeg: number;
  /** Mean pitch (deg, negative = looking down). Default −4. */
  pitchDeg: number;
  /** UI rotation of the frames. Default 90 (landscape). */
  screenAngle: ScreenAngle;
  /** Extra hand-held shake (tremor 8–12 Hz, fast pans). Default null (off). */
  shake: Partial<ShakeOptions> | null;
}

export const DEFAULT_YARD_TRAJECTORY: YardTrajectoryOptions = {
  center: [0, 0],
  semiAxes: [7.5, 5],
  laps: 1.03,
  duration: 40,
  height: 1.5,
  bobAmplitude: 0.03,
  stepHz: 2,
  swayAmplitude: 0.02,
  lateralOffset: 0,
  panAmplitudeDeg: 25,
  pitchDeg: -4,
  screenAngle: 90,
  shake: null,
};

/** Harmonic perturbation of the yard loop (fraction of each semi-axis). */
const YARD_H1 = 0.08,
  YARD_H2 = 0.08;

/** Point of the yard walking loop (centre line, no gait sway) at loop parameter s (rad). */
export function yardLoopPoint(s: number, opts: Partial<YardTrajectoryOptions> = {}): [number, number] {
  const o = { ...DEFAULT_YARD_TRAJECTORY, ...opts };
  const [a, b] = o.semiAxes;
  return [
    o.center[0] + a * Math.cos(s) + YARD_H1 * a * Math.cos(2 * s + 0.5),
    o.center[1] + b * Math.sin(s) + YARD_H2 * b * Math.sin(3 * s + 0.2),
  ];
}

export function createYardTrajectory(opts: Partial<YardTrajectoryOptions> = {}): Trajectory {
  const o = { ...DEFAULT_YARD_TRAJECTORY, ...opts };
  const T = o.duration;
  const [a, b] = o.semiAxes;
  const h1 = YARD_H1 * a,
    h2 = YARD_H2 * b;
  const Om = (2 * Math.PI * o.laps) / T;
  const deg = Math.PI / 180;
  const fn: JetTrajectoryFn = (t) => {
    // Loop parameter with a ±5 % surge at the step frequency.
    const s = jsum(jscale(t, Om), jsinusoid(t, (0.05 * Om) / (2 * Math.PI * o.stepHz), o.stepHz, 0));
    const s2 = jaddc(jscale(s, 2), 0.5),
      s3 = jaddc(jscale(s, 3), 0.2);
    const cx = jadd(jscale(jcos(s), a), jscale(jcos(s2), h1));
    const cy = jadd(jscale(jsin(s), b), jscale(jsin(s3), h2));
    // Tangent dc/ds and the unit lateral direction.
    const tx = jsub(jscale(jsin(s), -a), jscale(jsin(s2), 2 * h1));
    const ty = jadd(jscale(jcos(s), b), jscale(jcos(s3), 3 * h2));
    const tn = jsqrt(jadd(jmul(tx, tx), jmul(ty, ty)));
    const nx = jdiv(ty, tn),
      ny = jdiv(jneg(tx), tn);
    const sway0 = jsinusoid(t, o.swayAmplitude, o.stepHz / 2, 0.3);
    const sway = o.lateralOffset ? jaddc(sway0, o.lateralOffset) : sway0;
    const bob = jadd(jsinusoid(t, o.bobAmplitude, o.stepHz, 0), jsinusoid(t, 0.03, 1 / 13, 1.0));
    let p: JetVec3 = [
      jsum(jconst(o.center[0]), cx, jmul(sway, nx)),
      jsum(jconst(o.center[1]), cy, jmul(sway, ny)),
      jaddc(bob, o.height),
    ];
    const shakeP = shakeOffset(t, o.shake);
    if (shakeP) p = [jadd(p[0], shakeP[0]), jadd(p[1], shakeP[1]), jadd(p[2], shakeP[2])];
    const pan = o.panAmplitudeDeg * deg;
    let yaw = jsum(
      jatan2(cy, cx), // outward from the loop centre
      jsinusoid(t, 0.78 * pan, 1 / 16, 0.4),
      jsinusoid(t, 0.27 * pan, 1 / 5.3, 1.1),
      jsinusoid(t, 0.012, o.stepHz / 2, 0.9),
    );
    let pitch = jsum(
      jconst(o.pitchDeg * deg),
      jsinusoid(t, 3.5 * deg, 1 / 11, 0.5),
      jsinusoid(t, 0.01, o.stepHz, 1.3),
    );
    let roll = jadd(jsinusoid(t, 0.025, o.stepHz / 2, 0.2), jsinusoid(t, 0.015, 1 / 8.5, 0));
    const sh = shakeAngles(t, o.shake);
    if (sh) {
      yaw = jadd(yaw, sh.yaw);
      pitch = jadd(pitch, sh.pitch);
      roll = jadd(roll, sh.roll);
    }
    const R_wd = deviceAttitudeJet(cameraToWorldJet(yaw, pitch, roll), o.screenAngle);
    return { p, R_wd };
  };
  return createJetTrajectory('yard', T, o.screenAngle, fn);
}

/** Angular-speed statistics |ω_dev| (rad/s) sampled every dt seconds over [t0, t1]. */
export function angularSpeedStats(traj: Trajectory, t0: number, t1: number, dt = 0.005): { max: number; p50: number; p90: number; p99: number; tMax: number } {
  const v: number[] = [];
  let max = 0,
    tMax = t0;
  for (let t = t0; t <= t1 + 1e-9; t += dt) {
    const w = traj.evaluate(t).omegaDev;
    const m = Math.hypot(w[0], w[1], w[2]);
    v.push(m);
    if (m > max) {
      max = m;
      tMax = t;
    }
  }
  v.sort((a, b) => a - b);
  const q = (p: number) => v[Math.min(v.length - 1, Math.floor(p * v.length))];
  return { max, p50: q(0.5), p90: q(0.9), p99: q(0.99), tMax };
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/** Device at rest with a fixed attitude. */
export function createStaticTrajectory(
  R_wd: ArrayLike<number>,
  position: ArrayLike<number>,
  screenAngle: ScreenAngle = 0,
  duration = 10,
): Trajectory {
  const Rc = jmatConst(R_wd);
  const p: JetVec3 = [jconst(position[0]), jconst(position[1]), jconst(position[2])];
  return createJetTrajectory('static', duration, screenAngle, () => ({ p, R_wd: Rc }));
}

/** Device attitude from a camera-to-world rotation: R_wd = R_wc · R_cam_device(θ). */
export function deviceAttitudeFromCamera(R_wc: ArrayLike<number>, screenAngle: ScreenAngle): Mat3 {
  return m3Mul(R_wc, rCamDevice(screenAngle));
}
