/**
 * Orientation experiment for the halo filter's sky test: run `removeDepthHalos` on a view
 * rotated by k clockwise quarter turns (as displayed; the `rotateRGBA90` convention) and rotate
 * the result back. This simulates frames that are not gravity-upright (frames rotated to a
 * session's canonical orientation, or landscape frames under the iPhone portrait lock;
 * src/sensors/frame-orientation.ts, docs/research/ios-camera.md).
 *
 * One clockwise turn: pixel (x, y) → (H − 1 − y, x); camera axes x' = −y, y' = x, z' = z
 * (normals and the up vector rotate the same way); intrinsics cx' = H − 1 − cy, cy' = cx.
 */
import type { CameraIntrinsics, RGBAImage } from '../../core/types';
import { rotateRGBA90 } from '../../features/image';
import { removeDepthHalos, type HaloFilterOptions } from '../../dense/depth-filter';
import type { DepthMap } from '../../dense/types';

function turnDepthMap(dm: DepthMap): DepthMap {
  const W = dm.width;
  const H = dm.height;
  const n = W * H;
  const out: DepthMap = { width: H, height: W, depth: new Float32Array(n), normal: new Float32Array(3 * n), cost: new Float32Array(n) };
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const j = x * H + (H - 1 - y);
      out.depth[j] = dm.depth[i];
      out.cost[j] = dm.cost[i];
      out.normal[3 * j] = -dm.normal[3 * i + 1];
      out.normal[3 * j + 1] = dm.normal[3 * i];
      out.normal[3 * j + 2] = dm.normal[3 * i + 2];
    }
  }
  return out;
}

function turnCam(c: CameraIntrinsics): CameraIntrinsics {
  return { ...c, width: c.height, height: c.width, cx: c.height - 1 - c.cy, cy: c.cx };
}

function turnVec(v: ArrayLike<number>): [number, number, number] {
  return [-v[1], v[0], v[2]];
}

/** k ∈ {0..3} clockwise quarter turns of a depth map (exact permutation; `rotate(·, 4 − k)` undoes it). */
export function rotateDepthMap(dm: DepthMap, k: number): DepthMap {
  let out = dm;
  for (let t = 0; t < (((k % 4) + 4) % 4); t++) out = turnDepthMap(out);
  return out;
}

/**
 * `removeDepthHalos` on the view rotated by k clockwise quarter turns, result rotated back.
 * `options.skyUp` (camera frame of the UNrotated view) is rotated with the view.
 */
export function removeDepthHalosRotated(dm: DepthMap, image: RGBAImage, cam: CameraIntrinsics, options: Partial<HaloFilterOptions>, k: number): DepthMap {
  const q = ((Math.round(k) % 4) + 4) % 4;
  if (q === 0) return removeDepthHalos(dm, image, cam, options);
  let c = cam;
  let up = options.skyUp ?? null;
  for (let t = 0; t < q; t++) {
    c = turnCam(c);
    if (up) up = turnVec(up);
  }
  const res = removeDepthHalos(rotateDepthMap(dm, q), rotateRGBA90(image, q), c, { ...options, skyUp: up });
  return rotateDepthMap(res, 4 - q);
}
