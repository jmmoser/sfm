/**
 * Scan-time estimate from the benchmark's unit costs (BenchRates) and the engine's quality
 * presets (src/engine/presets.ts: feature working size / count, live pair budgets, the
 * resolved dense preset incl. DENSE_PRESETS). A ROUGH model — every coefficient below is an
 * assumption listed in `ESTIMATE_ASSUMPTIONS` and shown with the results:
 *
 *  - frame (live):  SIFT at the preset's working size (4:3) + guided matching to the last keyframe
 *                   (≈ half a brute-force pair) + 15 ms of JS (two-view RANSAC, colours, sharpness);
 *  - keyframe:      the live pair budget (sequential + retrieval + spatial) brute-force matched +
 *                   local BA (10 LM iterations over ≈ 15 images);
 *  - refine:        ≈ 12 new pairs per keyframe matched; incremental SfM ≈ 10 local-BA iterations
 *                   per image + a 10-iteration global BA every 10 images (average size K/2) + a
 *                   30-iteration final global BA; BA cost ∝ observations (≈ 0.35 × max features
 *                   per image);
 *  - dense:         views = min(K, maxViews); PatchMatch units = MP × sources × (photometric +
 *                   geometric passes × iterations) per view at the GPU size (CPU: cpuMaxImageSize);
 *                   fusion + halo filter ≈ 0.6 µs per pixel per view on the reference x86 core,
 *                   scaled by this device's CPU SIFT speed relative to it;
 *  - mesh:          TSDF ≈ 0.3 µs per pixel per view + 1.5 s marching cubes / simplification
 *                   (same CPU scaling).
 *
 * The coefficients are unverified on Apple devices (headless reference timings:
 * docs/research/on-device-benchmark.md).
 */
import { QUALITY_PRESETS, resolveDensePreset } from '../../engine/presets';
import type { BenchEstimate, BenchRates } from './types';

/** CPU SIFT speed (ms per MP, 4 cores x86, Node) of the machine the fusion constant was measured on. */
export const REF_CPU_SIFT_MS_PER_MP = 240;

export const ESTIMATE_ASSUMPTIONS: readonly string[] = [
  'a scan of {K} keyframes, 4:3 frames',
  'live matching to the last keyframe ≈ ½ brute-force pair; +15 ms JS per frame',
  'keyframe mapping: live pair budget brute-force matched + 10 local-BA iterations over 15 images',
  'refine: 12 new pairs per keyframe; incremental SfM ≈ 10 local-BA iterations per image + periodic and final global BA; 0.35 × max features observations per image',
  'dense: views = min(keyframes, preset max views); PatchMatch ∝ MP × sources × iterations; fusion 0.6 µs/px/view on the reference CPU, scaled by the CPU SIFT speed',
  'mesh: TSDF 0.3 µs/px/view + 1.5 s, CPU-scaled',
];

const QUALITIES = ['fast', 'balanced', 'max'] as const;

export interface EstimateOptions {
  /** Assumed number of keyframes of a scan (default 80). */
  keyframes?: number;
}

/** Estimates per quality preset; [] when a needed measurement is missing. */
export function estimateScanTimes(r: BenchRates, opts: EstimateOptions = {}): BenchEstimate[] {
  const K = Math.max(2, Math.round(opts.keyframes ?? 80));
  const featGpu = r.siftGpuMsPerMP !== null && r.matchGpuMsPerMPairs !== null;
  const siftRate = featGpu ? r.siftGpuMsPerMP : r.siftCpuMsPerMP;
  const matchRate = featGpu ? r.matchGpuMsPerMPairs : r.matchCpuMsPerMPairs;
  const denseGpu = r.pmGpuMsPerUnit !== null;
  const pmRate = denseGpu ? r.pmGpuMsPerUnit : r.pmCpuMsPerUnit;
  const ba = r.baMsPerIterPerKObs;
  if (siftRate === null || matchRate === null || pmRate === null || ba === null) return [];
  // This device's CPU relative to the reference machine (> 1 = slower).
  const cpuFactor = r.siftCpuMsPerMP !== null && r.siftCpuMsPerMP > 0 ? r.siftCpuMsPerMP / REF_CPU_SIFT_MS_PER_MP : 1;
  const out: BenchEstimate[] = [];
  for (const q of QUALITIES) {
    const p = QUALITY_PRESETS[q];
    const d = resolveDensePreset(q);
    const featMP = (p.featureMaxSide * Math.round(p.featureMaxSide * 0.75)) / 1e6;
    const pairMs = ((p.maxFeatures * p.maxFeatures) / 1e6) * matchRate;
    const siftMs = featMP * siftRate;
    const frameMs = siftMs + 0.5 * pairMs + 15;
    const obsPerImage = 0.35 * p.maxFeatures;
    const baIter = (images: number): number => ((images * obsPerImage) / 1000) * ba;
    const livePairs = p.liveSequentialPairs + p.liveRetrievalPairs + p.liveSpatialPairs;
    const keyframeMs = livePairs * pairMs + 10 * baIter(Math.min(K, 15));
    const refineMatch = K * 12 * pairMs;
    const incremental = K * 10 * baIter(Math.min(K, 15)) + Math.floor(K / 10) * 10 * baIter(K / 2) + 30 * baIter(K);
    const refineMs = refineMatch + incremental;
    const views = Math.min(K, d.maxViews);
    const side = denseGpu ? d.maxImageSize : d.cpuMaxImageSize;
    const denseMP = (side * Math.round(side * 0.75)) / 1e6;
    const its = d.iterations + d.geometricPasses * d.geometricIterations;
    const pmMs = views * denseMP * d.numSources * its * pmRate;
    const fusionMs = views * denseMP * 1e6 * 0.6e-3 * cpuFactor;
    const denseMs = pmMs + fusionMs;
    const meshMs = (views * denseMP * 1e6 * 0.3e-3 + 1500) * cpuFactor;
    out.push({
      quality: q,
      keyframes: K,
      frameMs: Math.round(frameMs),
      captureFps: Math.round(Math.min(10, 1000 / frameMs) * 10) / 10,
      keyframeMs: Math.round(keyframeMs),
      refineMs: Math.round(refineMs),
      denseMs: Math.round(denseMs),
      meshMs: Math.round(meshMs),
      processingMs: Math.round(refineMs + denseMs + meshMs),
      features: featGpu ? 'gpu' : 'cpu',
      dense: denseGpu ? 'gpu' : 'cpu',
    });
  }
  return out;
}

/** The assumptions with the scan size filled in. */
export function estimateAssumptions(keyframes = 80): string[] {
  return ESTIMATE_ASSUMPTIONS.map((a) => a.replace('{K}', String(keyframes)));
}
