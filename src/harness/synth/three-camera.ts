/**
 * OpenCV pose / intrinsics → three.js camera.
 *
 * three.js cameras look down −Z with +Y up; the OpenCV camera frame looks down +Z
 * with +Y down, so x_three = F x_cv with F = diag(1, −1, −1). The scene is rendered
 * directly in the Z-up world frame (we set the camera matrices explicitly, so three's
 * Y-up convention never enters).
 */
import { Matrix4, type Camera } from 'three';
import type { Pose } from '../../core/types';
import type { NormalizedWindow } from './conventions';

/**
 * Row-major 4x4 camera-to-world matrix in three's camera convention:
 * [ Rᵀ F | C ], C = −Rᵀ t.
 */
export function threeCameraWorldMatrix(pose: Pose, out: number[] = new Array<number>(16)): number[] {
  const R = pose.R,
    t = pose.t;
  const cx = -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]);
  const cy = -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]);
  const cz = -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]);
  // Rᵀ columns are the camera axes in world; multiply columns 2,3 by −1 (F).
  out[0] = R[0]; out[1] = -R[3]; out[2] = -R[6]; out[3] = cx;
  out[4] = R[1]; out[5] = -R[4]; out[6] = -R[7]; out[7] = cy;
  out[8] = R[2]; out[9] = -R[5]; out[10] = -R[8]; out[11] = cz;
  out[12] = 0; out[13] = 0; out[14] = 0; out[15] = 1;
  return out;
}

const _m = new Array<number>(16);

/** Set a three camera's world matrix from an OpenCV world-to-camera pose. */
export function applyPoseToThreeCamera(camera: Camera, pose: Pose): void {
  const m = threeCameraWorldMatrix(pose, _m);
  camera.matrixAutoUpdate = false;
  camera.matrix.set(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
  camera.matrixWorldNeedsUpdate = true;
  camera.updateMatrixWorld(true); // copies matrix → matrixWorld and updates matrixWorldInverse
}

/**
 * Off-axis perspective projection whose frustum is exactly the given window of
 * undistorted normalized coordinates (OpenCV y down): NDC x = −1 ↔ x = xmin,
 * NDC y = +1 (top) ↔ y = ymin.
 */
export function setWindowProjection(camera: Camera, w: NormalizedWindow, near: number, far: number): void {
  const left = w.xmin * near;
  const right = w.xmax * near;
  const top = -w.ymin * near;
  const bottom = -w.ymax * near;
  camera.projectionMatrix.makePerspective(left, right, top, bottom, near, far);
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
}

/** Convenience for tests: project a world point with a configured three camera to NDC. */
export function projectWithThreeCamera(camera: Camera, X: ArrayLike<number>): [number, number, number] {
  const m = new Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  const e = m.elements; // column-major
  const x = X[0],
    y = X[1],
    z = X[2];
  const cx = e[0] * x + e[4] * y + e[8] * z + e[12];
  const cy = e[1] * x + e[5] * y + e[9] * z + e[13];
  const cz = e[2] * x + e[6] * y + e[10] * z + e[14];
  const cw = e[3] * x + e[7] * y + e[11] * z + e[15];
  return [cx / cw, cy / cw, cz / cw];
}
