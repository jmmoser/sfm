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
 *
 * Every reply carries the request id and the job's compute time; failures reply `error`.
 */
import type { CameraIntrinsics, FeatureSet, MatchPairs } from '../core/types';
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
  | { id: number; op: 'rsCost'; key: number; data?: RSBundleCostData; model: RollingShutterModel };

export type CpuWorkerResponse = { id: number; ms: number; result?: unknown; error?: string };

interface WorkerScope {
  onmessage: ((ev: MessageEvent<CpuWorkerRequest>) => void) | null;
  postMessage(msg: CpuWorkerResponse, transfer?: Transferable[]): void;
}

const scope = self as unknown as WorkerScope;
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
    } else {
      throw new Error(`unknown op ${(req as { op?: unknown }).op}`);
    }
  } catch (e) {
    scope.postMessage({ id: (req as { id: number }).id, ms: performance.now() - t0, error: (e as Error)?.message ?? String(e) });
  }
};
