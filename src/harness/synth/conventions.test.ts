import { Camera } from 'three';
import { describe, expect, it } from 'vitest';
import type { CameraIntrinsics } from '../../core/types';
import {
  cameraCenter,
  deviceFromPose,
  distortNormalized,
  headingDegOfDirection,
  pinholeWindow,
  poseFromDevice,
  projectRadial,
  undistortNormalized,
  undistortedWindow,
} from './conventions';
import { expSO3, m3T, m3Vec, orthonormalityError } from './mat';
import { applyPoseToThreeCamera, projectWithThreeCamera, setWindowProjection } from './three-camera';
import { createObjectTrajectory, createYardTrajectory, trajectoryPose } from './trajectory';
import { simulateHeadings } from './heading';

const cam: CameraIntrinsics = { width: 960, height: 720, f: 753.4, cx: 478.5, cy: 361.0, k1: -0.03, k2: 0.005 };

describe('pose conventions', () => {
  it('poseFromDevice / deviceFromPose round trip; R = R_cam_device R_wdᵀ, C = position', () => {
    for (const theta of [0, 90, 180, 270]) {
      const R_wd = expSO3([0.3, -1.2, 0.7]);
      const C = [1.5, -2, 0.7];
      const pose = poseFromDevice(R_wd, C, theta);
      expect(orthonormalityError(pose.R)).toBeLessThan(1e-14);
      const c = cameraCenter(pose);
      for (let i = 0; i < 3; i++) expect(c[i]).toBeCloseTo(C[i], 13);
      // x_cam = R X + t maps the centre to the origin
      const x = m3Vec(pose.R, C);
      for (let i = 0; i < 3; i++) expect(x[i] + pose.t[i]).toBeCloseTo(0, 13);
      const back = deviceFromPose(pose, theta);
      for (let i = 0; i < 9; i++) expect(back.R_wd[i]).toBeCloseTo(R_wd[i], 14);
    }
  });

  it('a device vector maps to the camera frame through R_cam_device', () => {
    const R_wd = expSO3([0.1, 0.2, 0.3]);
    const pose = poseFromDevice(R_wd, [0, 0, 0], 90);
    // world vector = R_wd v_dev; camera vector = R (R_wd v_dev) must equal R_cam_device v_dev
    const vDev = [0.3, -0.5, 0.8];
    const vCam = m3Vec(pose.R, m3Vec(R_wd, vDev));
    expect(Array.from(vCam).map((v) => +v.toFixed(12))).toEqual([0.5, -0.3, -0.8].map((v) => +v.toFixed(12)));
  });

  it('trajectory frames keep the camera upright in the image (image +y points down in the world)', () => {
    for (const traj of [createObjectTrajectory(), createYardTrajectory()]) {
      for (const t of [0, 5, 10, 20]) {
        const p = trajectoryPose(traj, t);
        // second row of R = camera +y (image down) in world coordinates
        expect(p.R[5]).toBeLessThan(-0.6);
      }
    }
  });
});

describe('RADIAL camera model', () => {
  it('undistortNormalized inverts distortNormalized', () => {
    const o = new Float64Array(2),
      u = new Float64Array(2);
    for (let i = 0; i < 200; i++) {
      const x = -0.8 + (1.6 * i) / 199,
        y = 0.6 * Math.sin(i);
      distortNormalized(cam.k1, cam.k2, x, y, o);
      undistortNormalized(cam.k1, cam.k2, o[0], o[1], u);
      expect(Math.abs(u[0] - x)).toBeLessThan(1e-13);
      expect(Math.abs(u[1] - y)).toBeLessThan(1e-13);
    }
  });

  it('undistortedWindow contains the undistortion of every border pixel', () => {
    const w = undistortedWindow(cam, 0);
    const u = new Float64Array(2);
    for (let i = 0; i <= 100; i++) {
      for (const [px, py] of [
        [-0.5 + (cam.width * i) / 100, -0.5],
        [-0.5 + (cam.width * i) / 100, cam.height - 0.5],
        [-0.5, -0.5 + (cam.height * i) / 100],
        [cam.width - 0.5, -0.5 + (cam.height * i) / 100],
      ]) {
        undistortNormalized(cam.k1, cam.k2, (px - cam.cx) / cam.f, (py - cam.cy) / cam.f, u);
        expect(u[0]).toBeGreaterThanOrEqual(w.xmin - 1e-12);
        expect(u[0]).toBeLessThanOrEqual(w.xmax + 1e-12);
        expect(u[1]).toBeGreaterThanOrEqual(w.ymin - 1e-12);
        expect(u[1]).toBeLessThanOrEqual(w.ymax + 1e-12);
      }
    }
    // barrel distortion (k1 < 0) ⇒ the undistorted window is larger than the pinhole one
    const p = pinholeWindow(cam);
    expect(w.xmax - w.xmin).toBeGreaterThan(p.xmax - p.xmin);
  });
});

describe('three.js camera from OpenCV pose + intrinsics', () => {
  it('reproduces the pinhole projection exactly (principal point honoured)', () => {
    const pin: CameraIntrinsics = { ...cam, k1: 0, k2: 0, cx: 470.25, cy: 368.75 };
    const pose = trajectoryPose(createYardTrajectory(), 7.3);
    const c = new Camera();
    applyPoseToThreeCamera(c, pose);
    setWindowProjection(c, pinholeWindow(pin), 0.1, 500);
    const C = cameraCenter(pose);
    const out = new Float64Array(3);
    let worst = 0;
    for (let i = 0; i < 50; i++) {
      // points in front of the camera
      const xc = [Math.sin(i) * 3, Math.cos(1.7 * i) * 2, 4 + (i % 7)];
      const X = m3Vec(m3T(pose.R), xc);
      for (let k = 0; k < 3; k++) X[k] += C[k];
      expect(projectRadial(pin, pose, X, out)).toBe(true);
      const [nx, ny, nz] = projectWithThreeCamera(c, X);
      const u = -0.5 + ((nx + 1) / 2) * pin.width;
      const v = -0.5 + ((1 - ny) / 2) * pin.height;
      worst = Math.max(worst, Math.abs(u - out[0]), Math.abs(v - out[1]));
      expect(nz).toBeGreaterThan(-1);
      expect(nz).toBeLessThan(1);
    }
    expect(worst).toBeLessThan(1e-6);
  });

  it('camera world matrix: three camera looks down −Z with +Y up = OpenCV −y', () => {
    const pose = poseFromDevice(expSO3([0.4, 0.1, -0.2]), [1, 2, 3], 0);
    const c = new Camera();
    applyPoseToThreeCamera(c, pose);
    const e = c.matrixWorld.elements; // column-major
    const zAxis = [e[8], e[9], e[10]];
    const yAxis = [e[4], e[5], e[6]];
    const fwd = [pose.R[6], pose.R[7], pose.R[8]];
    const down = [pose.R[3], pose.R[4], pose.R[5]];
    for (let i = 0; i < 3; i++) {
      expect(zAxis[i]).toBeCloseTo(-fwd[i], 12);
      expect(yAxis[i]).toBeCloseTo(-down[i], 12);
    }
    expect(e[12]).toBeCloseTo(1, 12);
    expect(e[13]).toBeCloseTo(2, 12);
    expect(e[14]).toBeCloseTo(3, 12);
  });
});

describe('compass heading', () => {
  it('degrees clockwise from north: N=0, E=90, S=180, W=270', () => {
    expect(headingDegOfDirection([0, 1, 0])).toBeCloseTo(0, 12);
    expect(headingDegOfDirection([1, 0, 0])).toBeCloseTo(90, 12);
    expect(headingDegOfDirection([0, -1, 0])).toBeCloseTo(180, 12);
    expect(headingDegOfDirection([-1, 0, 0.5])).toBeCloseTo(270, 12);
    expect(Number.isNaN(headingDegOfDirection([0, 0, 1]))).toBe(true);
  });

  it('simulated headings follow the camera optical axis; noise σ ≈ 5°', () => {
    const traj = createYardTrajectory();
    const clean = simulateHeadings(traj, 0, 40, 1000, { sigmaDeg: 0 });
    for (const h of clean.slice(0, 50)) {
      const pose = trajectoryPose(traj, (h.t - 1000) / 1000);
      expect(h.heading).toBeCloseTo(headingDegOfDirection([pose.R[6], pose.R[7], pose.R[8]]), 9);
      expect(h.headingDeviceTop).not.toBeNull(); // landscape: device top is horizontal
    }
    const noisy = simulateHeadings(traj, 0, 40, 1000, {});
    const d = noisy.map((h) => ((h.heading - h.trueHeading + 540) % 360) - 180);
    const sd = Math.sqrt(d.reduce((a, b) => a + b * b, 0) / d.length);
    expect(sd).toBeGreaterThan(4.5);
    expect(sd).toBeLessThan(5.5);
    expect(noisy[0].accuracy).toBe(5);
  });
});
