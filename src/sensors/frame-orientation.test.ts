import { describe, expect, it } from 'vitest';
import {
  ScreenAngleTracker,
  canonicalRotation,
  cameraFromDevice,
  deviceFromCamera,
  deviceToCamera,
  quarterTurnMat3,
  rotateImagePoint,
  rotatedSize,
  screenAngleFrom,
  snapScreenAngle,
  uiUpInDevice,
  type ScreenAngle,
} from './frame-orientation';

const THETAS: ScreenAngle[] = [0, 90, 180, 270];

function mul(a: ArrayLike<number>, b: ArrayLike<number>): number[] {
  const o = new Array(9).fill(0);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) for (let k = 0; k < 3; k++) o[3 * r + c] += a[3 * r + k] * b[3 * k + c];
  return o;
}
function mulVec(a: ArrayLike<number>, v: ArrayLike<number>): number[] {
  return [0, 1, 2].map((r) => a[3 * r] * v[0] + a[3 * r + 1] * v[1] + a[3 * r + 2] * v[2]);
}
function transpose(a: ArrayLike<number>): number[] {
  return [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
}
function det(a: ArrayLike<number>): number {
  return a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
}
function rz(deg: number): number[] {
  const r = (deg * Math.PI) / 180;
  const c = Math.cos(r);
  const s = Math.sin(r);
  return [c, -s, 0, s, c, 0, 0, 0, 1];
}
function expectClose(a: ArrayLike<number>, b: ArrayLike<number>, tol = 1e-12): void {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol);
}

describe('screen angle', () => {
  it('snaps and maps iOS window.orientation (−90 → 270)', () => {
    expect(snapScreenAngle(-90)).toBe(270);
    expect(snapScreenAngle(90)).toBe(90);
    expect(snapScreenAngle(180)).toBe(180);
    expect(snapScreenAngle(-180)).toBe(180);
    expect(snapScreenAngle(360)).toBe(0);
    expect(snapScreenAngle(88)).toBe(90);
    expect(snapScreenAngle(NaN)).toBe(0);
  });
  it('prefers window.orientation over screen.orientation.angle (iPad reports (90−θ) mod 360)', () => {
    expect(screenAngleFrom({ windowOrientation: 90, screenOrientationAngle: 0 })).toBe(90); // iPad landscape
    expect(screenAngleFrom({ windowOrientation: -90, screenOrientationAngle: 180 })).toBe(270);
    expect(screenAngleFrom({ windowOrientation: 0, screenOrientationAngle: 90 })).toBe(0); // iPad portrait
    expect(screenAngleFrom({ windowOrientation: null, screenOrientationAngle: 270 })).toBe(270); // desktop
    expect(screenAngleFrom({})).toBe(0);
  });
});

describe('cameraFromDevice (ARCHITECTURE §2)', () => {
  it('is orthonormal with det +1 and equals diag(1,−1,−1)·Rz(θ) for all θ', () => {
    for (const th of THETAS) {
      const R = cameraFromDevice(th);
      expectClose(mul(R, transpose(R)), [1, 0, 0, 0, 1, 0, 0, 0, 1]);
      expect(det(R)).toBeCloseTo(1, 12);
      expectClose(R, mul([1, 0, 0, 0, -1, 0, 0, 0, -1], rz(th)));
      expectClose(deviceFromCamera(th), transpose(R));
      for (const v of R) expect(Number.isInteger(v)).toBe(true); // exact for multiples of 90°
    }
  });
  it('matches the documented rows for θ = 0, 90, 180, 270', () => {
    expect(Array.from(cameraFromDevice(0))).toEqual([1, 0, 0, 0, -1, 0, 0, 0, -1]);
    expect(Array.from(cameraFromDevice(90))).toEqual([0, -1, 0, -1, 0, 0, 0, 0, -1]);
    expect(Array.from(cameraFromDevice(180))).toEqual([-1, 0, 0, 0, 1, 0, 0, 0, -1]);
    expect(Array.from(cameraFromDevice(270))).toEqual([0, 1, 0, 1, 0, 0, 0, 0, -1]);
  });
  it('maps gravity of an upright-held phone to image-down (+y_cam) for every UI rotation', () => {
    // Held upright for UI rotation θ: the UI "up" axis points against gravity.
    for (const th of THETAS) {
      const up = uiUpInDevice(th);
      const gDev = [-up[0], -up[1], -up[2]];
      const gCam = deviceToCamera(gDev, th);
      expectClose(gCam, [0, 1, 0]);
      expectClose(gCam, mulVec(cameraFromDevice(th), gDev));
    }
  });
  it('optical axis is −z_device (rear camera)', () => {
    for (const th of THETAS) expectClose(mulVec(cameraFromDevice(th), [0, 0, -1]), [0, 0, 1]);
  });
  it('θ=90 (top edge left): image right = device bottom, image down = device left', () => {
    const R = cameraFromDevice(90);
    // device −y (bottom edge) → camera +x; device −x (left edge) → camera +y
    expectClose(mulVec(R, [0, -1, 0]), [1, 0, 0]);
    expectClose(mulVec(R, [-1, 0, 0]), [0, 1, 0]);
  });
});

describe('canonicalRotation', () => {
  it('returns clockwise quarter turns k = (θ_f − θ_s)/90 mod 4', () => {
    expect(canonicalRotation(0, 0)).toBe(0);
    expect(canonicalRotation(0, 90)).toBe(1);
    expect(canonicalRotation(0, 270)).toBe(3);
    expect(canonicalRotation(90, 0)).toBe(3);
    expect(canonicalRotation(270, 90)).toBe(2);
    expect(canonicalRotation(0, -90)).toBe(3);
  });
  it('quarterTurnMat3(k)·R_cd(θ_f) = R_cd(θ_s) for all pairs', () => {
    for (const s of THETAS) {
      for (const f of THETAS) {
        const k = canonicalRotation(s, f);
        expectClose(mul(quarterTurnMat3(k), cameraFromDevice(f)), cameraFromDevice(s));
      }
    }
  });
  it('pixel map agrees with the camera rotation (and with rotateRGBA90: clockwise)', () => {
    const W = 8;
    const H = 6;
    const f = 5;
    const cx = (W - 1) / 2;
    const cy = (H - 1) / 2;
    for (let k = 0; k < 4; k++) {
      const { width: W2, height: H2 } = rotatedSize(W, H, k);
      const cx2 = (W2 - 1) / 2;
      const cy2 = (H2 - 1) / 2;
      const Q = quarterTurnMat3(k);
      for (const [x, y] of [
        [0, 0],
        [7, 0],
        [3.5, 2],
        [1, 5],
      ]) {
        const p = rotateImagePoint(x, y, W, H, k);
        const ray = mulVec(Q, [(x - cx) / f, (y - cy) / f, 1]);
        expect(p[0]).toBeCloseTo(cx2 + (f * ray[0]) / ray[2], 12);
        expect(p[1]).toBeCloseTo(cy2 + (f * ray[1]) / ray[2], 12);
      }
    }
    // k=1 clockwise: top-left pixel goes to the top-right corner.
    expect(Array.from(rotateImagePoint(0, 0, W, H, 1))).toEqual([H - 1, 0]);
    expect(rotatedSize(W, H, 1)).toEqual({ width: H, height: W });
  });
  it('example: portrait session, landscape frame (top edge left) → device top ends up as image up', () => {
    const k = canonicalRotation(0, 90);
    // device top (+y) in the landscape frame's camera coordinates:
    const topCam = mulVec(cameraFromDevice(90), [0, 1, 0]);
    expectClose(topCam, [-1, 0, 0]); // points left in the frame
    expectClose(mulVec(quarterTurnMat3(k), topCam), [0, -1, 0]); // image up after rotation
  });
});

describe('ScreenAngleTracker', () => {
  it('records changes and settles after settleMs', () => {
    let angle: ScreenAngle = 0;
    let now = 1000;
    const tr = new ScreenAngleTracker({ read: () => angle, now: () => now, settleMs: 500 });
    expect(tr.angle).toBe(0);
    expect(tr.isSettled()).toBe(true);
    angle = 90;
    now = 2000;
    const seen: number[] = [];
    tr.onChange((a) => seen.push(a));
    expect(tr.angle).toBe(90);
    expect(seen).toEqual([90]);
    expect(tr.lastChangeTime).toBe(2000);
    expect(tr.isSettled(2300)).toBe(false);
    expect(tr.isSettled(2500)).toBe(true);
    expect(tr.angleAt(1500)).toBe(0);
    expect(tr.angleAt(2100)).toBe(90);
  });
});
