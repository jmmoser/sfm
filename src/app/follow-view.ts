/**
 * Point-cloud capture view (CaptureConfig.cloudView): the live 3D viewer's camera follows the phone.
 *
 * The anchor is the newest posed frame: every tracked frame's pose in the live model (the engine's
 * frame PnP), or a keyframe's when that is newer, with its capture time on the IMU clock. Until the
 * next one the gyro carries its rotation to now:
 *   R_view = R_cd(θ_ui) · G · R_cd(engine)ᵀ · R_k,   G = R_{d(now)←d(t_k)}   (src/sensors/motion.ts)
 * where R_cd(θ_ui) = cameraFromDevice(θ_ui) is the rear camera upright for the current UI rotation,
 * i.e. the screen's x right / y down / z forward frame the viewer expects. The position is the
 * anchor's centre (updated with every tracked frame).
 *
 * G is integrated incrementally (only the samples since the last view update). A new anchor (or a
 * model update that moves it) would make the view jump: the old and the new anchor are compared at
 * the SAME gyro time, so only the difference of the two models — not the phone's motion — is kept as
 * a correction that decays to zero (time constant TAU_MS). The gyro motion itself is never lagged.
 */
import type { Mat3, Pose, Vec3 } from '../core/types';
import { mat3Mul, mat3MulTranspose, mat3MulVec, mat3TransposeMul } from '../math/mat3';
import { mat3ToRotvec, rotvecToMat3 } from '../math/rotation';
import { cameraFromDevice, type ScreenAngle } from '../sensors/frame-orientation';

/** Time constant of the decay of an anchor jump (ms). */
const TAU_MS = 150;

/**
 * Device rotation R_{d(t1)←d(t0)} from the gyro (t1 < t0: the inverse), null when the IMU does not
 * cover [t0, t1].
 */
export type GyroRotation = (t0: number, t1: number) => Mat3 | null;

export interface FollowAnchor {
  /** World-to-camera pose of the keyframe (engine camera frame). */
  pose: Pose;
  /** Capture time on the IMU clock (ms); undefined = unknown (no gyro between keyframes). */
  t?: number;
}

/** World-to-view rotation now (see the file header); G null = no gyro (the keyframe's rotation). */
export function followViewRotation(Rk: ArrayLike<number>, R_cam_device: ArrayLike<number> | null, G: ArrayLike<number> | null, theta: ScreenAngle): Mat3 {
  const Rui = cameraFromDevice(theta);
  // engine camera frame unknown: assume the UI's (upright frames)
  const Rcd = R_cam_device ?? Rui;
  const dev = mat3TransposeMul(Rcd, Rk); // world → device at t_k
  return mat3Mul(Rui, G ? mat3Mul(G, dev) : dev);
}

/**
 * Vertical field of view (degrees) of a camera image shown `object-fit: cover` in a viewW × viewH
 * box; `swap` = the image is shown turned a quarter (UI rotated against the engine's frames).
 */
export function coverFovDeg(intr: { width: number; height: number; f: number }, swap: boolean, viewW: number, viewH: number): number {
  const iw = swap ? intr.height : intr.width;
  const ih = swap ? intr.width : intr.height;
  if (!(iw > 0 && ih > 0 && intr.f > 0 && viewW > 0 && viewH > 0)) return NaN;
  const scale = Math.max(viewW / iw, viewH / ih);
  return (2 * Math.atan(viewH / scale / 2 / intr.f) * 180) / Math.PI;
}

/** The engine's frames are shown a quarter turn off in UI rotation θ (from its R_cam_device). */
export function quarterTurned(R_cam_device: ArrayLike<number> | null, theta: ScreenAngle): boolean {
  if (!R_cam_device) return false;
  // R_cd(θ) = [[c, −s, 0], [−s, −c, 0], [0, 0, −1]]
  const sessionTheta = Math.round((Math.atan2(-R_cam_device[1], R_cam_device[0]) * 180) / Math.PI / 90) * 90;
  return (((sessionTheta - theta) / 90) % 2 + 2) % 2 === 1;
}

/** Follows the newest keyframe + gyro, with anchor jumps smoothed out. */
export class FollowView {
  private anchor: FollowAnchor | null = null;
  private Rcd: Mat3 | null = null;
  /** Gyro rotation since the anchor's time, up to gyro time tG (null = no gyro for this anchor). */
  private G: Mat3 | null = null;
  private tG = NaN;
  /** A new anchor waiting to replace the current one at the next pose() (same gyro time). */
  private pending: { anchor: FollowAnchor | null; Rcd: Mat3 | null } | null = null;
  /** A centre jump larger than this (world units) is a re-initialised model: shown at once. */
  private jumpLimit = Infinity;
  /** Decaying jump corrections: rotation (left, view frame) and centre offset (world). */
  private corrRot: Vec3 = new Float64Array(3);
  private corrPos: Vec3 = new Float64Array(3);
  private lastNow = NaN;

  private readonly gyro: GyroRotation;

  constructor(gyro: GyroRotation) {
    this.gyro = gyro;
  }

  get hasAnchor(): boolean {
    return (this.pending ? this.pending.anchor : this.anchor) !== null;
  }

  /**
   * New live model: the newest camera becomes the anchor. `jumpLimit` (world units): a centre jump
   * larger than this is a re-initialised model and is shown at once, not smoothed.
   */
  setAnchor(anchor: FollowAnchor | null, R_cam_device: ArrayLike<number> | null = null, jumpLimit = Infinity): void {
    this.pending = { anchor, Rcd: R_cam_device ? Float64Array.from(R_cam_device) : null };
    this.jumpLimit = jumpLimit;
  }

  /** Forget the smoothing state (the view was hidden): the next pose is shown as is. */
  reset(): void {
    this.corrRot.fill(0);
    this.corrPos.fill(0);
    this.lastNow = NaN;
  }

  /**
   * World-to-view pose at `now` (ms, for the smoothing) with the gyro integrated up to `tGyro` (IMU
   * clock; NaN = no gyro: the rotation stays where the gyro last left it).
   */
  pose(now: number, tGyro: number, theta: ScreenAngle): Pose | null {
    const dt = Number.isFinite(this.lastNow) ? Math.max(0, now - this.lastNow) : 0;
    this.lastNow = now;
    const k = Math.exp(-dt / TAU_MS);
    for (let i = 0; i < 3; i++) {
      this.corrRot[i] *= k;
      this.corrPos[i] *= k;
    }
    this.advance(tGyro);
    const p = this.pending;
    if (p) {
      this.pending = null;
      const before = this.anchor ? this.view(theta) : null;
      this.anchor = p.anchor;
      this.Rcd = p.Rcd;
      // the new anchor's gyro rotation up to the same time tG as the old one
      const ta = p.anchor?.t;
      this.G = ta !== undefined && Number.isFinite(ta) && Number.isFinite(this.tG) ? this.gyro(ta, this.tG) : null;
      const after = this.anchor ? this.view(theta) : null;
      if (before && after) {
        // keep showing what was shown: correction = shown · newᵀ, centre offset = shown − new
        const shown = mat3Mul(rotvecToMat3(this.corrRot), before.R);
        const corr = mat3ToRotvec(mat3MulTranspose(shown, after.R));
        const dc = [0, 1, 2].map((i) => before.C[i] + this.corrPos[i] - after.C[i]);
        // a jump this large is a re-initialised model, not drift: show it at once
        if (Math.hypot(corr[0], corr[1], corr[2]) < Math.PI / 4 && Math.hypot(dc[0], dc[1], dc[2]) <= this.jumpLimit) {
          this.corrRot.set(corr);
          this.corrPos.set(dc);
        } else this.reset();
      }
    }
    if (!this.anchor) return null;
    const { R, C } = this.view(theta);
    const Rs = mat3Mul(rotvecToMat3(this.corrRot), R);
    const Cs = new Float64Array([C[0] + this.corrPos[0], C[1] + this.corrPos[1], C[2] + this.corrPos[2]]);
    const t = mat3MulVec(Rs, Cs);
    return { R: Rs, t: new Float64Array([-t[0], -t[1], -t[2]]) };
  }

  /** Extend G with the gyro samples up to tGyro (only the new ones). */
  private advance(tGyro: number): void {
    if (!Number.isFinite(tGyro) || !(tGyro > this.tG) || !this.anchor) {
      if (!this.anchor && Number.isFinite(tGyro)) this.tG = tGyro;
      return;
    }
    if (this.G) {
      const d = this.gyro(this.tG, tGyro);
      // a gap in the IMU (sensors restarted): start over from the anchor's time
      this.G = d ? mat3Mul(d, this.G) : this.anchor.t !== undefined ? this.gyro(this.anchor.t, tGyro) : null;
    } else if (this.anchor.t !== undefined && Number.isFinite(this.anchor.t)) this.G = this.gyro(this.anchor.t, tGyro);
    this.tG = tGyro;
  }

  /** Raw view (no correction) of the current anchor: world-to-view rotation and centre. */
  private view(theta: ScreenAngle): { R: Mat3; C: Vec3 } {
    const a = this.anchor!;
    const R = followViewRotation(a.pose.R, this.Rcd, this.G, theta);
    const Rk = a.pose.R,
      tk = a.pose.t;
    const C = new Float64Array([-(Rk[0] * tk[0] + Rk[3] * tk[1] + Rk[6] * tk[2]), -(Rk[1] * tk[0] + Rk[4] * tk[1] + Rk[7] * tk[2]), -(Rk[2] * tk[0] + Rk[5] * tk[1] + Rk[8] * tk[2])]);
    return { R, C };
  }
}

/** Median distance of (a sample of) the points from c: the scene depth, the scale of a re-init jump. */
export function medianDistance(positions: ArrayLike<number>, c: ArrayLike<number>, maxSamples = 512): number {
  const n = Math.floor(positions.length / 3);
  if (!n) return NaN;
  const step = Math.max(1, Math.floor(n / maxSamples));
  const d: number[] = [];
  for (let i = 0; i < n; i += step) d.push(Math.hypot(positions[3 * i] - c[0], positions[3 * i + 1] - c[1], positions[3 * i + 2] - c[2]));
  d.sort((a, b) => a - b);
  return d[d.length >> 1];
}

/**
 * Per-point splat radius (world units) so the sparse cloud reads as surfaces: 0.5 × the distance to
 * the nearest neighbour (hash grid of `cell`), clamped to [rMin, cell]; isolated points get rMin.
 */
export function splatRadii(positions: ArrayLike<number>, cell: number, rMin: number): Float32Array {
  const n = Math.floor(positions.length / 3);
  const out = new Float32Array(n);
  if (!(cell > 0)) return out.fill(rMin);
  const inv = 1 / cell;
  const key = (x: number, y: number, z: number): number => ((x * 73856093) ^ (y * 19349663) ^ (z * 83492791)) | 0;
  const grid = new Map<number, number[]>();
  const ix = new Int32Array(n),
    iy = new Int32Array(n),
    iz = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    ix[i] = Math.floor(positions[3 * i] * inv);
    iy[i] = Math.floor(positions[3 * i + 1] * inv);
    iz[i] = Math.floor(positions[3 * i + 2] * inv);
    const k = key(ix[i], iy[i], iz[i]);
    const c = grid.get(k);
    if (c) c.push(i);
    else grid.set(k, [i]);
  }
  for (let i = 0; i < n; i++) {
    const px = positions[3 * i],
      py = positions[3 * i + 1],
      pz = positions[3 * i + 2];
    let best = cell * cell;
    for (let dx = -1; dx <= 1; dx++)
      for (let dy = -1; dy <= 1; dy++)
        for (let dz = -1; dz <= 1; dz++) {
          const c = grid.get(key(ix[i] + dx, iy[i] + dy, iz[i] + dz));
          if (!c) continue;
          for (const j of c) {
            if (j === i) continue;
            const ex = positions[3 * j] - px,
              ey = positions[3 * j + 1] - py,
              ez = positions[3 * j + 2] - pz;
            const d2 = ex * ex + ey * ey + ez * ez;
            if (d2 < best) best = d2;
          }
        }
    out[i] = best < cell * cell ? Math.min(cell, Math.max(rMin, 0.5 * Math.sqrt(best))) : rMin;
  }
  return out;
}
