/**
 * Camera frustums + trajectory overlay. All geometry is in the WORLD frame (it
 * lives under the viewer's world root, which converts to three's Y-up frame).
 */
import {
  BufferAttribute,
  BufferGeometry,
  Color,
  Group,
  Line,
  LineBasicMaterial,
  LineSegments,
  type ColorRepresentation,
} from 'three';
import type { CameraIntrinsics, Pose } from '../core/types';
import { cameraCenter } from './helpers';

export interface ViewerCamera {
  pose: Pose;
  intrinsics: CameraIntrinsics;
}

/** Line segments per frustum: 4 rays, 4 rectangle edges, 2 for the "up" marker. */
export const SEGMENTS_PER_CAMERA = 10;
export const VERTICES_PER_CAMERA = 2 * SEGMENTS_PER_CAMERA;

/**
 * World-frame corners of the image rectangle at depth `depth` in front of the
 * camera (pinhole, distortion ignored), order: top-left, top-right,
 * bottom-right, bottom-left (image convention, pixel edges at −0.5 / W−0.5).
 * Writes 12 numbers into `out` at `offset`.
 */
export function frustumCorners(
  pose: Pose,
  cam: CameraIntrinsics,
  depth: number,
  out: Float64Array | Float32Array = new Float64Array(12),
  offset = 0,
): Float64Array | Float32Array {
  const C = cameraCenter(pose);
  const R = pose.R;
  const us = [-0.5, cam.width - 0.5, cam.width - 0.5, -0.5];
  const vs = [-0.5, -0.5, cam.height - 0.5, cam.height - 0.5];
  for (let k = 0; k < 4; k++) {
    const x = ((us[k] - cam.cx) / cam.f) * depth;
    const y = ((vs[k] - cam.cy) / cam.f) * depth;
    const z = depth;
    // X = Rᵀ x_cam + C
    out[offset + 3 * k] = R[0] * x + R[3] * y + R[6] * z + C[0];
    out[offset + 3 * k + 1] = R[1] * x + R[4] * y + R[7] * z + C[1];
    out[offset + 3 * k + 2] = R[2] * x + R[5] * y + R[8] * z + C[2];
  }
  return out;
}

const _corners = new Float64Array(12);
const _c = new Float64Array(3);

/**
 * Fill `positions` (and `colors`, linear RGB floats) with the frustum line
 * segments of all cameras. Returns the number of vertices written
 * (= VERTICES_PER_CAMERA · cams.length). Arrays must be large enough.
 */
export function writeFrustumLines(
  cams: ArrayLike<ViewerCamera>,
  depth: number,
  highlightLast: boolean,
  positions: Float32Array,
  colors: Float32Array,
  normalColor: [number, number, number],
  highlightColor: [number, number, number],
): number {
  let v = 0;
  const put = (x: number, y: number, z: number, col: [number, number, number]) => {
    positions[3 * v] = x;
    positions[3 * v + 1] = y;
    positions[3 * v + 2] = z;
    colors[3 * v] = col[0];
    colors[3 * v + 1] = col[1];
    colors[3 * v + 2] = col[2];
    v++;
  };
  for (let i = 0; i < cams.length; i++) {
    const { pose, intrinsics: cam } = cams[i];
    const hl = highlightLast && i === cams.length - 1;
    const col = hl ? highlightColor : normalColor;
    const d = hl ? depth * 1.6 : depth;
    frustumCorners(pose, cam, d, _corners, 0);
    cameraCenter(pose, _c);
    const k = _corners;
    // rays
    for (let j = 0; j < 4; j++) {
      put(_c[0], _c[1], _c[2], col);
      put(k[3 * j], k[3 * j + 1], k[3 * j + 2], col);
    }
    // rectangle
    for (let j = 0; j < 4; j++) {
      const n = (j + 1) % 4;
      put(k[3 * j], k[3 * j + 1], k[3 * j + 2], col);
      put(k[3 * n], k[3 * n + 1], k[3 * n + 2], col);
    }
    // "up" triangle above the top edge (image −y): base at 25 % / 75 % of the top edge.
    const tl = 0;
    const tr = 3;
    const bl = 9;
    const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
    const b0 = [0, 1, 2].map((a) => lerp(k[tl + a], k[tr + a], 0.25));
    const b1 = [0, 1, 2].map((a) => lerp(k[tl + a], k[tr + a], 0.75));
    // apex: middle of top edge moved further "up" by 35 % of the image height
    const apex = [0, 1, 2].map((a) => lerp(k[tl + a], k[tr + a], 0.5) + (k[tl + a] - k[bl + a]) * 0.35);
    put(b0[0], b0[1], b0[2], col);
    put(apex[0], apex[1], apex[2], col);
    put(apex[0], apex[1], apex[2], col);
    put(b1[0], b1[1], b1[2], col);
  }
  return v;
}

function linearRGB(c: ColorRepresentation): [number, number, number] {
  const col = new Color(c);
  return [col.r, col.g, col.b];
}

/** three.js objects for the camera overlay (frustums + trajectory polyline). */
export class CameraOverlay {
  readonly object = new Group();
  private frustumGeom = new BufferGeometry();
  private trajGeom = new BufferGeometry();
  private readonly frustumMat = new LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 });
  private readonly trajMat: LineBasicMaterial;
  private readonly frustums: LineSegments;
  private readonly trajectory: Line;
  private capacity = 0; // cameras
  private positions = new Float32Array(0);
  private colors = new Float32Array(0);
  private trajPositions = new Float32Array(0);
  private readonly normal: [number, number, number];
  private readonly highlight: [number, number, number];
  count = 0;

  constructor(colors: { frustum?: ColorRepresentation; highlight?: ColorRepresentation; trajectory?: ColorRepresentation } = {}) {
    this.normal = linearRGB(colors.frustum ?? '#5ab4ff');
    this.highlight = linearRGB(colors.highlight ?? '#ffa21f');
    this.trajMat = new LineBasicMaterial({ color: colors.trajectory ?? '#c8d6e5', transparent: true, opacity: 0.6 });
    this.frustums = new LineSegments(this.frustumGeom, this.frustumMat);
    this.trajectory = new Line(this.trajGeom, this.trajMat);
    this.frustums.frustumCulled = false;
    this.trajectory.frustumCulled = false;
    this.object.add(this.frustums, this.trajectory);
    this.object.name = 'cameras';
    this.frustums.visible = false;
    this.trajectory.visible = false;
  }

  private ensureCapacity(n: number): void {
    if (n <= this.capacity) return;
    const cap = Math.max(16, Math.ceil(n * 1.5));
    this.capacity = cap;
    this.positions = new Float32Array(3 * VERTICES_PER_CAMERA * cap);
    this.colors = new Float32Array(3 * VERTICES_PER_CAMERA * cap);
    this.trajPositions = new Float32Array(3 * cap);
    // New attributes need new geometries (three cannot resize GPU buffers in place).
    this.frustumGeom.dispose();
    this.trajGeom.dispose();
    this.frustumGeom = new BufferGeometry();
    this.trajGeom = new BufferGeometry();
    this.frustumGeom.setAttribute('position', new BufferAttribute(this.positions, 3));
    this.frustumGeom.setAttribute('color', new BufferAttribute(this.colors, 3));
    this.trajGeom.setAttribute('position', new BufferAttribute(this.trajPositions, 3));
    this.frustums.geometry = this.frustumGeom;
    this.trajectory.geometry = this.trajGeom;
  }

  /** Rebuild geometry. `depth` = frustum depth in world units. */
  update(cams: ArrayLike<ViewerCamera>, depth: number, highlightLast: boolean): void {
    const n = cams.length;
    this.count = n;
    this.ensureCapacity(n);
    const nv = writeFrustumLines(cams, depth, highlightLast, this.positions, this.colors, this.normal, this.highlight);
    for (let i = 0; i < n; i++) {
      cameraCenter(cams[i].pose, _c);
      this.trajPositions[3 * i] = _c[0];
      this.trajPositions[3 * i + 1] = _c[1];
      this.trajPositions[3 * i + 2] = _c[2];
    }
    const pa = this.frustumGeom.getAttribute('position') as BufferAttribute | undefined;
    const ca = this.frustumGeom.getAttribute('color') as BufferAttribute | undefined;
    const ta = this.trajGeom.getAttribute('position') as BufferAttribute | undefined;
    if (pa && ca && ta) {
      pa.clearUpdateRanges();
      ca.clearUpdateRanges();
      ta.clearUpdateRanges();
      pa.addUpdateRange(0, 3 * nv);
      ca.addUpdateRange(0, 3 * nv);
      ta.addUpdateRange(0, 3 * n);
      pa.needsUpdate = true;
      ca.needsUpdate = true;
      ta.needsUpdate = true;
    }
    this.frustumGeom.setDrawRange(0, nv);
    this.trajGeom.setDrawRange(0, n);
    this.frustums.visible = n > 0;
    this.trajectory.visible = n >= 2;
  }

  dispose(): void {
    this.frustumGeom.dispose();
    this.trajGeom.dispose();
    this.frustumMat.dispose();
    this.trajMat.dispose();
  }
}
