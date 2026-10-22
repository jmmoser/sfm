/**
 * CPU helper worker of the engine (nested in the engine worker; src/engine/cpu-pool.ts): pure,
 * deterministic CPU jobs that refine runs for many independent inputs.
 *
 *  - `verify`: two-view verification of putative matches (`estimateTwoViewGeometry`, the same
 *    code and RANSAC seed as on the engine thread, so the result is identical).
 *  - `codebook` / `quantize`: visual-word quantisation of a keyframe's descriptors with a
 *    vocabulary (`VocabularyIndex.quantize`, exact), the codebook sent once per key.
 *  - `rsCost`: robust cost of a rolling-shutter hypothesis (`RSBundleCost.cost`, a short subset
 *    BA), the cost function's data sent with its first request on this worker.
 *  - `denseView` (densify): a stored keyframe decoded (createImageBitmap + OffscreenCanvas, as on
 *    the engine thread), undistorted and rolling-shutter corrected, + its colour thumbnail
 *    (src/engine/dense-jobs.ts `decodeDenseView`, `denseColourThumbnail`); images transferred back.
 *  - `haloStore` (densify): halo / sky mask of a final depth map applied to the worker's copy and
 *    its float16 serialisation (`haloStoreJob`); mask and serialised map transferred back.
 *
 * Every reply carries the request id and the job's compute time; failures reply `error`.
 */
import type { CameraIntrinsics, FeatureSet, MatchPairs, RGBAImage } from '../core/types';
import type { DepthMap } from '../dense/types';
import { decodeDenseView, denseColourThumbnail, haloStoreJob, type DenseViewRS, type HaloJobInput } from '../engine/dense-jobs';
import type { StoredImage } from '../engine/persistence';
import { estimateTwoViewGeometry, type TwoViewOptions } from '../geometry/twoview';
import { RSBundleCost, type RollingShutterModel, type RSBundleCostData } from '../sfm/rolling-shutter';
import { VocabularyIndex, type VocabularyOptions } from '../sfm/vocab';

export type CpuWorkerRequest =
  | {
      id: number;
      op: 'verify';
      kp1: Float32Array;
      kp2: Float32Array;
      matches: MatchPairs;
      cam1: CameraIntrinsics;
      cam2: CameraIntrinsics;
      opts: TwoViewOptions;
    }
  | { id: number; op: 'codebook'; key: number; codebook: Float64Array; options: Partial<VocabularyOptions> }
  | { id: number; op: 'quantize'; key: number; count: number; descriptors: Uint8Array; scores: Float32Array }
  | { id: number; op: 'rsCost'; key: number; data?: RSBundleCostData; model: RollingShutterModel }
  | { id: number; op: 'denseView'; stored: StoredImage; distorted: CameraIntrinsics; maxSize: number; rs: DenseViewRS | null; thumbBytes: number }
  | { id: number; op: 'haloStore'; map: DepthMap; halo: HaloJobInput | null; serialize: boolean };

export type CpuWorkerResponse = { id: number; ms: number; result?: unknown; error?: string };

interface WorkerScope {
  onmessage: ((ev: MessageEvent<CpuWorkerRequest>) => void) | null;
  postMessage(msg: CpuWorkerResponse, transfer?: Transferable[]): void;
}

const scope = self as unknown as WorkerScope;

/** Distinct buffers of images (a thumbnail may be the image itself). */
function imageBuffers(...imgs: (RGBAImage | null)[]): Transferable[] {
  const out = new Set<ArrayBufferLike>();
  for (const im of imgs) if (im) out.add(im.data.buffer);
  return [...out] as Transferable[];
}
const vocabularies = new Map<number, VocabularyIndex>();
/** The rolling-shutter cost functions of the running estimation (one per pass; the older ones are dropped). */
const rsCosts = new Map<number, RSBundleCost>();

scope.onmessage = (ev) => {
  const req = ev.data;
  const t0 = performance.now();
  try {
    if (req.op === 'verify') {
      const g = estimateTwoViewGeometry(req.kp1, req.kp2, req.matches, req.cam1, req.cam2, req.opts);
      scope.postMessage({ id: req.id, ms: performance.now() - t0, result: g }, [g.inliers.buffer]);
    } else if (req.op === 'codebook') {
      vocabularies.set(req.key, new VocabularyIndex(req.codebook, req.options));
      scope.postMessage({ id: req.id, ms: performance.now() - t0 });
    } else if (req.op === 'quantize') {
      const v = vocabularies.get(req.key);
      if (!v) throw new Error(`no codebook ${req.key}`);
      const fs = { count: req.count, descriptors: req.descriptors, scores: req.scores } as FeatureSet;
      const words = v.quantize(fs);
      scope.postMessage({ id: req.id, ms: performance.now() - t0, result: words }, [words.buffer]);
    } else if (req.op === 'rsCost') {
      if (req.data) {
        for (const k of [...rsCosts.keys()]) if (k < req.key) rsCosts.delete(k);
        rsCosts.set(req.key, RSBundleCost.fromData(req.data));
      }
      const bc = rsCosts.get(req.key);
      if (!bc) throw new Error(`no rolling-shutter cost ${req.key}`);
      scope.postMessage({ id: req.id, ms: performance.now() - t0, result: bc.cost(req.model) });
    } else if (req.op === 'denseView') {
      // asynchronous (the decode): replies when done, failures included
      void (async () => {
        try {
          const v = await decodeDenseView(req.stored, req.distorted, req.maxSize, req.rs);
          const thumb = req.thumbBytes > 0 ? denseColourThumbnail(v.image, req.thumbBytes) : null;
          scope.postMessage({ id: req.id, ms: performance.now() - t0, result: { image: v.image, cam: v.cam, thumb } }, imageBuffers(v.image, thumb));
        } catch (e) {
          scope.postMessage({ id: req.id, ms: performance.now() - t0, error: (e as Error)?.message ?? String(e) });
        }
      })();
    } else if (req.op === 'haloStore') {
      const r = haloStoreJob(req.map, req.halo, req.serialize);
      const transfer: Transferable[] = [];
      if (r.mask) transfer.push(r.mask.buffer);
      if (r.stored) transfer.push(r.stored);
      scope.postMessage({ id: req.id, ms: performance.now() - t0, result: r }, transfer);
    } else {
      throw new Error(`unknown op ${(req as { op?: unknown }).op}`);
    }
  } catch (e) {
    scope.postMessage({ id: (req as { id: number }).id, ms: performance.now() - t0, error: (e as Error)?.message ?? String(e) });
  }
};
