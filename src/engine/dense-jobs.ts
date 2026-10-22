/**
 * CPU steps of the dense depth-map stage that run either on the engine thread or on a CPU helper
 * worker (src/workers/cpu.worker.ts through src/engine/cpu-pool.ts), the same code in both places
 * so the results are bit-identical:
 *
 *  - dense view preparation: a stored keyframe decoded at the undistorted size, undistorted and
 *    rolling-shutter corrected (`decodeDenseView` = what `loadDenseViewImage` in dense.ts does),
 *    plus the colour thumbnail fusion / TSDF sample (`denseColourThumbnail`);
 *  - the final maps' clean-up and store copy: halo / sky mask (`depthHaloMask`) applied to the map
 *    and its float16 serialisation for IndexedDB (`haloStoreJob`).
 *
 * `DenseHelpers` is the interface runDensify uses for the worker side (CpuPool implements it; tests
 * inject fakes). Plain data only: everything crosses a postMessage.
 */
import type { CameraIntrinsics, RGBAImage } from '../core/types';
import { applyDepthHaloMask, depthHaloMask, type HaloFilterOptions } from '../dense/depth-filter';
import type { DepthMap } from '../dense/types';
import { undistortImage, undistortedCamera } from '../dense/undistort';
import { serializeDepthMap } from '../io/serialize';
import { rsCorrectPinholeImage } from '../sfm/rolling-shutter';
import { fitSize, resizeRGBA } from './frame-ingest';
import { decodeStoredImage } from './image-cache';
import type { StoredImage } from './persistence';
import type { SparseModel } from './types';

/** Rolling-shutter correction of one dense view (`SparseModel.rollingShutter` at that image). */
export interface DenseViewRS {
  axis: 'rows' | 'cols';
  readoutMs: number;
  omega: [number, number, number];
  nu: [number, number, number] | null;
}

/** The view's rolling-shutter correction, or null (the model was not corrected / no angular velocity for it). */
export function denseViewRS(model: SparseModel, id: number): DenseViewRS | null {
  const rs = model.rollingShutter;
  const w = rs ? rs.omega[id] : undefined;
  if (!rs || !w) return null;
  return { axis: rs.axis, readoutMs: rs.readoutMs, omega: w, nu: rs.nu?.[id] ?? null };
}

/** Undistorted (≤ maxSize) and, with `rs`, rolling-shutter corrected pinhole image of a decoded keyframe. */
export function correctDenseView(rgba: RGBAImage, distorted: CameraIntrinsics, maxSize: number, rs: DenseViewRS | null): { image: RGBAImage; cam: CameraIntrinsics } {
  const und = undistortImage(rgba, distorted, maxSize);
  return { cam: und.cam, image: rs ? rsCorrectPinholeImage(und.image, und.cam, rs.omega, rs, rs.nu) : und.image };
}

/** A stored keyframe decoded at the undistorted size and corrected (`correctDenseView`). */
export async function decodeDenseView(stored: StoredImage, distorted: CameraIntrinsics, maxSize: number, rs: DenseViewRS | null): Promise<{ image: RGBAImage; cam: CameraIntrinsics }> {
  const c = undistortedCamera(distorted, maxSize);
  return correctDenseView(await decodeStoredImage(stored, c.width, c.height), distorted, maxSize, rs);
}

/**
 * Colour thumbnail of a dense view within `bytes` (area-averaged, long side ≥ 64 px; the image
 * itself when it fits): fusion / TSDF sample it at the projection with the frame-stamp pose.
 */
export function denseColourThumbnail(image: RGBAImage, bytes: number): RGBAImage {
  const w = image.width,
    h = image.height;
  const s = Math.min(1, Math.sqrt(bytes / (4 * w * h)));
  const fs = fitSize(w, h, Math.max(64, Math.floor(s * Math.max(w, h))));
  return resizeRGBA(image, fs.width, fs.height);
}

/** Inputs of the halo filter of one final depth map (`removeDepthHalos(map, image, cam, options)`). */
export interface HaloJobInput {
  image: RGBAImage;
  cam: CameraIntrinsics;
  options: Partial<HaloFilterOptions>;
}

export interface HaloStoreResult {
  /** Pixels the halo filter rejects (`depthHaloMask`), null = none or no filter. */
  mask: Uint32Array | null;
  /** The halo filter threw (the map is kept unfiltered, as on the engine thread). */
  haloError?: string;
  /** `serializeDepthMap(filtered map, { float16: true })` when requested (the IndexedDB copy). */
  stored?: ArrayBuffer;
}

/**
 * Halo / sky mask of `map` (with `halo`) applied to it IN PLACE, then its float16 serialisation
 * (with `serialize`). On the engine thread's copy of the map, `applyDepthHaloMask(map, mask)` then
 * gives exactly `removeDepthHalos(map, …)`.
 */
export function haloStoreJob(map: DepthMap, halo: HaloJobInput | null, serialize: boolean): HaloStoreResult {
  let mask: Uint32Array | null = null;
  let haloError: string | undefined;
  if (halo) {
    try {
      mask = depthHaloMask(map, halo.image, halo.cam, halo.options);
      if (mask) applyDepthHaloMask(map, mask);
    } catch (e) {
      mask = null;
      haloError = (e as Error)?.message ?? String(e);
    }
  }
  const out: HaloStoreResult = { mask };
  if (haloError !== undefined) out.haloError = haloError;
  if (serialize) out.stored = serializeDepthMap(map, { float16: true });
  return out;
}

/** CPU helper jobs of the depth-map stage (CpuPool; the work runs on another thread). */
export interface DenseHelpers {
  /** Usable workers. */
  readonly size: number;
  /** Compute time reported by the workers (ms). */
  readonly busyMs: { denseView: number; haloStore: number };
  /** `decodeDenseView` (+ `denseColourThumbnail` when `thumbBytes` > 0); `thumb` may be `image` itself. */
  denseView(
    stored: StoredImage,
    distorted: CameraIntrinsics,
    maxSize: number,
    rs: DenseViewRS | null,
    thumbBytes: number,
  ): Promise<{ image: RGBAImage; cam: CameraIntrinsics; thumb: RGBAImage | null }>;
  /** `haloStoreJob` on a COPY of `map` (the caller's map is untouched). */
  haloStore(map: DepthMap, halo: HaloJobInput | null, serialize: boolean): Promise<HaloStoreResult>;
  dispose(): void;
}
