/**
 * Screen (UI) orientation θ and the rear-camera ← device rotation (ARCHITECTURE §2).
 *
 * θ = how far the device is turned COUNTER-CLOCKWISE from natural portrait, as seen by
 * the user looking at the screen, θ ∈ {0, 90, 180, 270}. Camera frames delivered by
 * `<video>` / `createImageBitmap(video)` are upright for the current UI orientation θ
 * (WebKit `AVVideoCaptureSource::computeVideoFrameRotation`, docs/research/ios-sensors.md §5).
 *
 * Verified mapping (docs/research/ios-sensors.md §5, WebKit WKWebViewIOS.mm):
 *  - `window.orientation` ∈ {0, 90, −90, 180}; 90 = UIInterfaceOrientationLandscapeRight =
 *    device turned counter-clockwise (top edge pointing LEFT); −90 = top edge pointing RIGHT.
 *    ⇒ θ = (window.orientation + 360) % 360.
 *  - iPhone `screen.orientation.angle` = (window.orientation + 360) % 360 (same thing), but on
 *    iPad WebKit treats the natural orientation as landscape and reports (90 − θ) mod 360.
 *    Therefore `window.orientation` is PREFERRED when present (iOS, Android Chrome/Firefox),
 *    and `screen.orientation.angle` is the fallback (desktop browsers).
 *
 * R_cam_device(θ) = diag(1, −1, −1) · Rz(θ)   (v_cam = R_cam_device · v_device)
 *   θ=0:   rows (1,0,0) (0,−1,0) (0,0,−1)
 *   θ=90:  rows (0,−1,0) (−1,0,0) (0,0,−1)
 *   θ=180: rows (−1,0,0) (0,1,0) (0,0,−1)
 *   θ=270: rows (0,1,0) (1,0,0) (0,0,−1)
 *
 * Canonical orientation: a capture session fixes θ_s (usually the first frame's θ). A frame
 * captured at θ_f is rotated by k = canonicalRotation(θ_s, θ_f) CLOCKWISE quarter turns (as
 * displayed — the convention of `rotateRGBA90` in src/features/image.ts) so that its image
 * axes match the session's; afterwards its R_cam_device is cameraFromDevice(θ_s)
 * = quarterTurnMat3(k) · cameraFromDevice(θ_f), and pixel coordinates map with
 * rotateImagePoint(x, y, W, H, k).
 */
import type { Mat3, Vec3 } from '../core/types';

export type ScreenAngle = 0 | 90 | 180 | 270;
export type QuarterTurns = 0 | 1 | 2 | 3;

/** Snap any angle in degrees to the nearest of 0, 90, 180, 270 (handles negatives, e.g. −90 → 270). */
export function snapScreenAngle(deg: number): ScreenAngle {
  if (!Number.isFinite(deg)) return 0;
  const q = ((Math.round(deg / 90) % 4) + 4) % 4;
  return (q * 90) as ScreenAngle;
}

/**
 * Pure version of {@link screenAngle}: θ from the raw platform values. `windowOrientation`
 * (iOS / Android, device-natural-portrait referenced) wins over `screenOrientationAngle`.
 */
export function screenAngleFrom(src: {
  windowOrientation?: number | null;
  screenOrientationAngle?: number | null;
}): ScreenAngle {
  const wo = src.windowOrientation;
  if (typeof wo === 'number' && Number.isFinite(wo)) return snapScreenAngle(wo);
  const sa = src.screenOrientationAngle;
  if (typeof sa === 'number' && Number.isFinite(sa)) return snapScreenAngle(sa);
  return 0;
}

/** Current UI rotation θ (see file header for the verified iOS mapping). 0 outside a browser. */
export function screenAngle(): ScreenAngle {
  const g = globalThis as unknown as {
    orientation?: unknown;
    screen?: { orientation?: { angle?: unknown } };
  };
  const wo = typeof g.orientation === 'number' ? g.orientation : null;
  const sa = typeof g.screen?.orientation?.angle === 'number' ? (g.screen.orientation.angle as number) : null;
  return screenAngleFrom({ windowOrientation: wo, screenOrientationAngle: sa });
}

function cosSinDeg(deg: number): [number, number] {
  const r = (deg * Math.PI) / 180;
  let c = Math.cos(r);
  let s = Math.sin(r);
  // Exact values for multiples of 90° (no 6e-17 residues).
  if (Math.abs(c) < 1e-12) c = 0;
  if (Math.abs(s) < 1e-12) s = 0;
  if (Math.abs(Math.abs(c) - 1) < 1e-12) c = Math.sign(c);
  if (Math.abs(Math.abs(s) - 1) < 1e-12) s = Math.sign(s);
  return [c, s];
}

/**
 * R_cam_device for the rear camera given the UI rotation θ of the video frames
 * (ARCHITECTURE §2): diag(1,−1,−1)·Rz(θ) = [[c, −s, 0], [−s, −c, 0], [0, 0, −1]].
 * Any angle is accepted (exact for multiples of 90°).
 */
export function cameraFromDevice(screenAngleDeg: number, out: Mat3 = new Float64Array(9)): Mat3 {
  const [c, s] = cosSinDeg(screenAngleDeg);
  out[0] = c;
  out[1] = 0 - s;
  out[2] = 0;
  out[3] = 0 - s;
  out[4] = 0 - c;
  out[5] = 0;
  out[6] = 0;
  out[7] = 0;
  out[8] = -1;
  return out;
}

/** R_device_cam = R_cam_deviceᵀ. */
export function deviceFromCamera(screenAngleDeg: number, out: Mat3 = new Float64Array(9)): Mat3 {
  const [c, s] = cosSinDeg(screenAngleDeg);
  // Symmetric apart from the sign pattern: transpose of [[c,−s,0],[−s,−c,0],[0,0,−1]] is itself.
  out[0] = c;
  out[1] = 0 - s;
  out[2] = 0;
  out[3] = 0 - s;
  out[4] = 0 - c;
  out[5] = 0;
  out[6] = 0;
  out[7] = 0;
  out[8] = -1;
  return out;
}

/**
 * Number of CLOCKWISE quarter turns (as displayed; `rotateRGBA90` convention) that bring a
 * frame captured at UI angle `frameTheta` into the session's canonical orientation
 * `sessionTheta`: k = ((θ_f − θ_s) / 90) mod 4.
 *
 * Derivation: R_cd(θ_s)·R_cd(θ_f)ᵀ = D·Rz(θ_s − θ_f)·D = Rz(θ_f − θ_s) with D = diag(1,−1,−1);
 * Rz(φ) in the image frame (x right, y down) turns +x towards +y, i.e. clockwise on screen.
 * Example: session portrait (0), frame in landscape with the top edge left (90): the device
 * top points left in the frame, a 90° clockwise turn moves it up → k = 1.
 */
export function canonicalRotation(sessionTheta: number, frameTheta: number): QuarterTurns {
  const s = snapScreenAngle(sessionTheta);
  const f = snapScreenAngle(frameTheta);
  return (((((f - s) / 90) % 4) + 4) % 4) as QuarterTurns;
}

/**
 * Camera-frame rotation applied by turning an image k quarter turns clockwise:
 * v_cam(rotated image) = quarterTurnMat3(k) · v_cam(original) = Rz(k·90°).
 * Hence cameraFromDevice(θ_s) = quarterTurnMat3(canonicalRotation(θ_s, θ_f)) · cameraFromDevice(θ_f),
 * and a pose R (world→camera) of the original frame becomes quarterTurnMat3(k)·R.
 */
export function quarterTurnMat3(k: number, out: Mat3 = new Float64Array(9)): Mat3 {
  const [c, s] = cosSinDeg((((Math.round(k) % 4) + 4) % 4) * 90);
  out[0] = c;
  out[1] = 0 - s;
  out[2] = 0;
  out[3] = s;
  out[4] = c;
  out[5] = 0;
  out[6] = 0;
  out[7] = 0;
  out[8] = 1;
  return out;
}

/** Image size after k clockwise quarter turns. */
export function rotatedSize(width: number, height: number, k: number): { width: number; height: number } {
  const q = ((Math.round(k) % 4) + 4) % 4;
  return q % 2 === 1 ? { width: height, height: width } : { width, height };
}

/**
 * Map a pixel position (pixel-centre-at-integer convention, ARCHITECTURE §2) of a W×H image
 * into the image rotated k quarter turns clockwise — identical to src/features rotateRGBA90:
 * k=1: (x, y) → (H−1−y, x); k=2: (W−1−x, H−1−y); k=3: (y, W−1−x).
 * Intrinsics follow the same map: cx' , cy' = rotateImagePoint(cx, cy, W, H, k), f unchanged.
 */
export function rotateImagePoint(
  x: number,
  y: number,
  width: number,
  height: number,
  k: number,
  out: Float64Array = new Float64Array(2),
): Float64Array {
  const q = ((Math.round(k) % 4) + 4) % 4;
  switch (q) {
    case 0:
      out[0] = x;
      out[1] = y;
      break;
    case 1:
      out[0] = height - 1 - y;
      out[1] = x;
      break;
    case 2:
      out[0] = width - 1 - x;
      out[1] = height - 1 - y;
      break;
    default:
      out[0] = y;
      out[1] = width - 1 - x;
      break;
  }
  return out;
}

/** Gravity (or any device-frame vector) expressed in the camera frame for UI angle θ. */
export function deviceToCamera(vDevice: ArrayLike<number>, screenAngleDeg: number, out: Vec3 = new Float64Array(3)): Vec3 {
  const [c, s] = cosSinDeg(screenAngleDeg);
  const x = vDevice[0];
  const y = vDevice[1];
  const z = vDevice[2];
  out[0] = c * x - s * y;
  out[1] = 0 - s * x - c * y;
  out[2] = 0 - z;
  return out;
}

/**
 * The UI "up" direction (top of the upright UI) in the device frame for UI angle θ:
 * Rz(−θ)·(0, 1, 0) = (sin θ, cos θ, 0). θ=90 (top edge left) ⇒ the right edge (+x) is up.
 */
export function uiUpInDevice(screenAngleDeg: number, out: Vec3 = new Float64Array(3)): Vec3 {
  const [c, s] = cosSinDeg(screenAngleDeg);
  out[0] = s;
  out[1] = c;
  out[2] = 0;
  return out;
}

type Listener = (angle: ScreenAngle, t: number) => void;

/**
 * Tracks θ over time (screen.orientation 'change', window 'orientationchange' and 'resize').
 * Frames within `settleMs` (default 500 ms) of a change should be dropped: Safari 26.x can
 * deliver transiently mis-rotated frames (fixed in 27.0, 180429147).
 */
export class ScreenAngleTracker {
  readonly settleMs: number;
  private current: ScreenAngle;
  private changeT = -Infinity;
  private readonly history: { t: number; angle: ScreenAngle }[] = [];
  private readonly listeners = new Set<Listener>();
  private readonly read: () => ScreenAngle;
  private readonly now: () => number;
  private started = false;
  private readonly onEvent = (): void => {
    this.poll();
  };

  constructor(opts: { settleMs?: number; read?: () => ScreenAngle; now?: () => number } = {}) {
    this.settleMs = opts.settleMs ?? 500;
    this.read = opts.read ?? screenAngle;
    this.now = opts.now ?? (() => performance.now());
    this.current = this.read();
    this.history.push({ t: -Infinity, angle: this.current });
  }

  start(): void {
    if (this.started) return;
    this.started = true;
    const g = globalThis as unknown as {
      addEventListener?: (n: string, f: () => void) => void;
      screen?: { orientation?: EventTarget };
    };
    g.screen?.orientation?.addEventListener?.('change', this.onEvent);
    g.addEventListener?.('orientationchange', this.onEvent);
    g.addEventListener?.('resize', this.onEvent);
    this.poll();
  }

  stop(): void {
    if (!this.started) return;
    this.started = false;
    const g = globalThis as unknown as {
      removeEventListener?: (n: string, f: () => void) => void;
      screen?: { orientation?: EventTarget };
    };
    g.screen?.orientation?.removeEventListener?.('change', this.onEvent);
    g.removeEventListener?.('orientationchange', this.onEvent);
    g.removeEventListener?.('resize', this.onEvent);
  }

  /** Re-read θ now (also called by every event); records a change with its time. */
  poll(t: number = this.now()): ScreenAngle {
    const a = this.read();
    if (a !== this.current) {
      this.current = a;
      this.changeT = t;
      this.history.push({ t, angle: a });
      if (this.history.length > 32) this.history.splice(0, this.history.length - 32);
      for (const cb of this.listeners) {
        try {
          cb(a, t);
        } catch (e) {
          console.error(e);
        }
      }
    }
    return this.current;
  }

  /** Current θ (polled, so a missed event is caught). */
  get angle(): ScreenAngle {
    return this.poll();
  }

  /** performance.now() time of the last θ change (−Infinity if none). */
  get lastChangeTime(): number {
    return this.changeT;
  }

  /** θ that was in effect at time t (from the change history). */
  angleAt(t: number): ScreenAngle {
    for (let i = this.history.length - 1; i >= 0; i--) {
      if (this.history[i].t <= t) return this.history[i].angle;
    }
    return this.history[0].angle;
  }

  /** True when no θ change happened within `settleMs` before `t`. */
  isSettled(t: number = this.now(), settleMs: number = this.settleMs): boolean {
    this.poll();
    return t - this.changeT >= settleMs;
  }

  onChange(cb: Listener): () => void {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  }
}
