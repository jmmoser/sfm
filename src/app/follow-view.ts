/**
 * Point-cloud capture view (CaptureConfig.cloudView): the live 3D viewer's camera follows the phone.
 *
 * The anchor is the newest registered keyframe (its world-to-camera pose from the live model and
 * its capture time on the IMU clock). Between keyframes the gyro carries its rotation to now:
 *   R_view = R_cd(θ_ui) · G · R_cd(engine)ᵀ · R_k,   G = R_{d(now)←d(t_k)}   (src/sensors/motion.ts)
 * where R_cd(θ_ui) = cameraFromDevice(θ_ui) is the rear camera upright for the current UI rotation,
 * i.e. the screen's x right / y down / z forward frame the viewer expects. The position stays at
 * the keyframe's centre: the engine estimates no pose for the frames between keyframes.
 *
 * A new anchor (or a model update that moves it) would make the view jump: the jump is kept as a
 * correction that decays to zero (time constant TAU_MS), so the gyro motion itself is never lagged.
 */
import type { Mat3, Pose, Vec3 } from '../core/types';
import { mat3Mul, mat3MulTranspose, mat3MulVec, mat3TransposeMul } from '../math/mat3';
import { mat3ToRotvec, rotvecToMat3 } from '../math/rotation';
import { cameraFromDevice, type ScreenAngle } from '../sensors/frame-orientation';

/** Time constant of the decay of an anchor jump (ms). */
const TAU_MS = 250;

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
  /** Last pose shown (world-to-view rotation, centre) and the decaying jump corrections. */
  private lastR: Mat3 | null = null;
  private lastC: Vec3 | null = null;
  private corrRot: Vec3 = new Float64Array(3);
  private corrPos: Vec3 = new Float64Array(3);
  private lastNow = NaN;
  private jumped = false;

  get hasAnchor(): boolean {
    return this.anchor !== null;
  }

  get anchorTime(): number | undefined {
    return this.anchor?.t;
  }

  /** New live model: the newest camera becomes the anchor. */
  setAnchor(anchor: FollowAnchor | null, R_cam_device: ArrayLike<number> | null = null): void {
    this.anchor = anchor;
    this.Rcd = R_cam_device ? Float64Array.from(R_cam_device) : null;
    this.jumped = true;
  }

  /** Forget the smoothing state (the view was hidden): the next pose is shown as is. */
  reset(): void {
    this.lastR = null;
    this.lastC = null;
    this.corrRot.fill(0);
    this.corrPos.fill(0);
    this.lastNow = NaN;
  }

  /** World-to-view pose at `now` (ms) given the gyro rotation since the anchor's time (or null). */
  pose(now: number, G: ArrayLike<number> | null, theta: ScreenAngle): Pose | null {
    const a = this.anchor;
    if (!a) return null;
    const R = followViewRotation(a.pose.R, this.Rcd, G, theta);
    const Rk = a.pose.R,
      tk = a.pose.t;
    const C = new Float64Array([-(Rk[0] * tk[0] + Rk[3] * tk[1] + Rk[6] * tk[2]), -(Rk[1] * tk[0] + Rk[4] * tk[1] + Rk[7] * tk[2]), -(Rk[2] * tk[0] + Rk[5] * tk[1] + Rk[8] * tk[2])]);
    const dt = Number.isFinite(this.lastNow) ? Math.max(0, now - this.lastNow) : 0;
    this.lastNow = now;
    const k = Math.exp(-dt / TAU_MS);
    for (let i = 0; i < 3; i++) {
      this.corrRot[i] *= k;
      this.corrPos[i] *= k;
    }
    if (this.jumped && this.lastR && this.lastC) {
      // keep showing the last pose: correction = shown · newᵀ (left, view frame), centre offset
      const shown = mat3Mul(rotvecToMat3(this.corrRot), this.lastR);
      const corr = mat3ToRotvec(mat3MulTranspose(shown, R));
      // a jump this large is a re-initialised model, not drift: show it at once
      if (Math.hypot(corr[0], corr[1], corr[2]) < Math.PI / 4) {
        this.corrRot.set(corr);
        for (let i = 0; i < 3; i++) this.corrPos[i] = this.lastC[i] + this.corrPos[i] - C[i];
      } else {
        this.corrRot.fill(0);
        this.corrPos.fill(0);
      }
    }
    this.jumped = false;
    this.lastR = R;
    this.lastC = C;
    const Rs = mat3Mul(rotvecToMat3(this.corrRot), R);
    const Cs = new Float64Array([C[0] + this.corrPos[0], C[1] + this.corrPos[1], C[2] + this.corrPos[2]]);
    const t = mat3MulVec(Rs, Cs);
    return { R: Rs, t: new Float64Array([-t[0], -t[1], -t[2]]) };
  }
}
