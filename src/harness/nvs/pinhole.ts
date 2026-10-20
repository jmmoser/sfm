/**
 * Pinhole test cameras of the novel-view evaluation and their three.js projection (pure; the
 * renderers in ./renderers.ts apply it to the app's Viewer).
 */

/** Pinhole test camera: world-to-camera pose (OpenCV axes, row-major R) and intrinsics in pixels (centres at integers). */
export interface NvsCamera {
  width: number;
  height: number;
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  /** Row-major 3 × 3 world-to-camera rotation (x_cam = R·X + t). */
  R: ArrayLike<number>;
  t: ArrayLike<number>;
}

/**
 * three.js perspective parameters reproducing a pinhole camera exactly: vertical fov from fy, the
 * aspect from fx / fy, and a view offset moving the optical axis from the frustum centre
 * ((W − 1) / 2, (H − 1) / 2 in pixel-centre coordinates) to (cx, cy).
 */
export function pinholePerspective(cam: Pick<NvsCamera, 'width' | 'height' | 'fx' | 'fy' | 'cx' | 'cy'>): { fovDeg: number; aspect: number; offsetX: number; offsetY: number } {
  const { width: W, height: H } = cam;
  return {
    fovDeg: (2 * Math.atan(H / (2 * cam.fy)) * 180) / Math.PI,
    aspect: (W * cam.fy) / (H * cam.fx),
    offsetX: (W - 1) / 2 - cam.cx,
    offsetY: (H - 1) / 2 - cam.cy,
  };
}

/** GL readback (rows bottom to top) → rows top to bottom. */
export function flipRows(data: Uint8Array, width: number, height: number): Uint8Array {
  const out = new Uint8Array(data.length);
  const row = 4 * width;
  for (let y = 0; y < height; y++) out.set(data.subarray((height - 1 - y) * row, (height - y) * row), y * row);
  return out;
}
