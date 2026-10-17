/**
 * Benchmark & self-test runner (runs inside bench.worker.ts; also in Node for the CPU path).
 *
 *  1. SIFT: WebGPU extraction of a procedural 1920×1440 image (stage timings: upload, detect,
 *     select, describe, assemble) + GPU vs CPU parity on a smaller image (keypoints within
 *     1 px / 20 % scale, descriptor cosine) and the CPU reference speed.
 *  2. Matching: GPU brute-force matcher on 4096 × 4096 synthetic RootSIFT descriptors (first call
 *     incl. upload, cached call) vs the CPU matcher (identical pairs expected) + recall against
 *     the known correspondences.
 *  3. PatchMatch: WebGPU stereo on a ray-traced synthetic scene (src/dense/synthetic) at 640×480
 *     with 8 sources (per-iteration timing, accuracy vs the ground-truth depth) + GPU vs CPU
 *     agreement on a small scene (also the CPU fallback's speed).
 *  4. Bundle adjustment: a synthetic 100-image problem (src/ba/synthetic), time per LM iteration.
 *  5. Scan-time estimates per quality preset from the unit costs (estimate.ts).
 *
 * Without WebGPU (older iOS, Lockdown Mode, `gpu: false`) the GPU parts are skipped and the CPU
 * fallbacks are measured instead — nothing here statically imports a WebGPU module (sift-gpu /
 * match-gpu read GPUBufferUsage at module load, which throws where WebGPU does not exist).
 * Cancellation is checked between steps (and per BA iteration); synchronous CPU steps finish
 * first — the UI terminates the worker if it does not stop in time.
 */
import { bundleAdjust } from '../../ba/solver';
import { fixGaugeTwoPoses, makeSyntheticBA, perturbBA, syntheticErrors } from '../../ba/synthetic';
import { DESCRIPTOR_SIZE, type FeatureSet, type GrayImage, type MatchPairs } from '../../core/types';
import { patchMatchCPU } from '../../dense/patchmatch-cpu';
import { evaluateDepthMap, makeSyntheticDataset, surfaceMask, type SynthDataset } from '../../dense/synthetic';
import type { DepthMap } from '../../dense/types';
import { matchCPU } from '../../features/match';
import { extractSiftCPU, lastSiftCPUTimings, releaseSiftCPUMemory } from '../../features/sift-cpu';
import { featureParity } from '../../features/sift-gpu-host';
import { TestRng, makeTexturedImage } from '../../features/testdata';
import type { GPUContext } from '../../gpu/device';
import { estimateAssumptions, estimateScanTimes } from './estimate';
import { SCALES } from './scales';
import { BENCH_TESTS, type BenchCheck, type BenchDevice, type BenchRates, type BenchReport, type BenchScale, type BenchTestId, type BenchTestResult } from './types';

export interface RunOptions {
  scale: BenchScale;
  /** Use WebGPU when available (false: CPU fallbacks only). */
  gpu: boolean;
  /** Why `gpu` is false (shown in the report). */
  noGpuReason?: string;
  cancel: { cancelled: boolean };
  onTest?: (r: BenchTestResult) => void;
  onProgress?: (id: BenchTestId, message: string) => void;
  /** Assumed scan size of the estimates (default 80 keyframes). */
  keyframes?: number;
}

/** SIFT options of every run: no automatic ×2 upsampling, so the cost is ∝ pixels (as at the presets' ≥ 1024 px). */
const SIFT_OPTS = { maxImageSize: 2048, maxFeatures: 4096, upsampleBelow: 0 } as const;

class Cancelled extends Error {
  constructor() {
    super('Cancelled');
    this.name = 'CancelledError';
  }
}

const now = (): number => performance.now();
const yieldTick = (): Promise<void> => new Promise((r) => setTimeout(r, 0));
const median = (a: number[]): number => {
  if (!a.length) return NaN;
  const s = [...a].sort((x, y) => x - y);
  return s.length % 2 ? s[s.length >> 1] : 0.5 * (s[s.length / 2 - 1] + s[s.length / 2]);
};
const r1 = (v: number): number => Math.round(v * 10) / 10;
const r3 = (v: number): number => Math.round(v * 1000) / 1000;
const pct = (v: number): string => `${(100 * v).toFixed(1)} %`;

function check(name: string, value: number, op: '>=' | '<=' | '<' | '>', threshold: number, fmt: (v: number) => string = (v) => String(r3(v))): BenchCheck {
  const pass = op === '>=' ? value >= threshold : op === '<=' ? value <= threshold : op === '<' ? value < threshold : value > threshold;
  const glyph = op === '>=' ? '≥' : op === '<=' ? '≤' : op;
  return { name, pass: Number.isFinite(value) && pass, detail: `${fmt(value)} (need ${glyph} ${fmt(threshold)})` };
}

function newResult(id: BenchTestId, gpu: boolean): BenchTestResult {
  const title = BENCH_TESTS.find((t) => t.id === id)!.title;
  // Bundle adjustment has no GPU path.
  return { id, title, status: 'running', backend: gpu && id !== 'ba' ? 'gpu' : 'cpu', summary: '', timings: {}, metrics: {}, checks: [], ms: 0 };
}

function finish(r: BenchTestResult, t0: number): BenchTestResult {
  r.ms = Math.round(now() - t0);
  if (r.status === 'running') r.status = r.error ? 'error' : r.checks.every((c) => c.pass) ? 'pass' : 'fail';
  return r;
}

function errMsg(e: unknown): string {
  return (e as Error)?.message ?? String(e);
}

/** Gray (0..1) → RGBA bytes (for an ImageBitmap of the test image). */
function grayToImageData(img: GrayImage): ImageData | null {
  if (typeof ImageData === 'undefined') return null;
  const d = new Uint8ClampedArray(img.width * img.height * 4);
  for (let i = 0; i < img.width * img.height; i++) {
    const v = Math.round((img.data[i] as number) * 255);
    d[4 * i] = v;
    d[4 * i + 1] = v;
    d[4 * i + 2] = v;
    d[4 * i + 3] = 255;
  }
  return new ImageData(d, img.width, img.height);
}

// ---------------------------------------------------------------------------------------------
// synthetic descriptors for the matcher test

interface MatchSets {
  a: FeatureSet;
  b: FeatureSet;
  /** b index → true a index (−1 for distractors). */
  truth: Int32Array;
}

function quantize(v: Float64Array, out: Uint8Array, o: number): void {
  let ss = 0;
  for (let k = 0; k < DESCRIPTOR_SIZE; k++) ss += v[k] * v[k];
  const inv = 1 / Math.sqrt(ss || 1);
  for (let k = 0; k < DESCRIPTOR_SIZE; k++) out[o + k] = Math.min(255, Math.round(512 * v[k] * inv));
}

/** n RootSIFT-like descriptors in a, and b = 60 % noisy copies of a (shuffled) + 40 % distractors. */
export function makeMatchSets(n: number, seed = 5): MatchSets {
  const rng = new TestRng(seed);
  const D = DESCRIPTOR_SIZE;
  const mk = (): FeatureSet => ({ count: n, keypoints: new Float32Array(4 * n), descriptors: new Uint8Array(D * n), scores: new Float32Array(n), width: 1920, height: 1440 });
  const a = mk();
  const b = mk();
  const va: Float64Array[] = [];
  const v = new Float64Array(D);
  for (let i = 0; i < n; i++) {
    const x = new Float64Array(D);
    for (let k = 0; k < D; k++) x[k] = rng.next() < 0.35 ? 0 : rng.next() ** 2;
    va.push(x);
    quantize(x, a.descriptors, i * D);
    a.keypoints.set([rng.next() * 1920, rng.next() * 1440, 1 + 4 * rng.next(), rng.next() * 6.28], 4 * i);
    a.scores[i] = 1 - i / n;
  }
  const truth = new Int32Array(n).fill(-1);
  const perm = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng.next() * (i + 1));
    [perm[i], perm[j]] = [perm[j], perm[i]];
  }
  const nTrue = Math.round(0.6 * n);
  for (let j = 0; j < n; j++) {
    if (j < nTrue) {
      const src = va[perm[j]];
      for (let k = 0; k < D; k++) v[k] = Math.max(0, src[k] + 0.06 * rng.gaussian());
      truth[j] = perm[j];
    } else for (let k = 0; k < D; k++) v[k] = rng.next() < 0.35 ? 0 : rng.next() ** 2;
    quantize(v, b.descriptors, j * D);
    b.keypoints.set([rng.next() * 1920, rng.next() * 1440, 1 + 4 * rng.next(), rng.next() * 6.28], 4 * j);
    b.scores[j] = 1 - j / n;
  }
  return { a, b, truth };
}

/** Fraction of the reference pairs reproduced (as MatcherGPU.selfTest). */
export function pairOverlap(ref: MatchPairs, test: MatchPairs): number {
  const s = new Set<number>();
  for (let k = 0; k < ref.length; k += 2) s.add(ref[k] * 4294967296 + ref[k + 1]);
  let common = 0;
  for (let k = 0; k < test.length; k += 2) if (s.has(test[k] * 4294967296 + test[k + 1])) common++;
  const n = Math.max(ref.length, test.length) / 2;
  return n ? common / n : 1;
}

function matchRecall(m: MatchPairs, truth: Int32Array): { recall: number; precision: number } {
  let correct = 0;
  for (let k = 0; k < m.length; k += 2) if (truth[m[k + 1]] === m[k]) correct++;
  let nTrue = 0;
  for (const t of truth) if (t >= 0) nTrue++;
  return { recall: nTrue ? correct / nTrue : 1, precision: m.length ? correct / (m.length / 2) : 1 };
}

// ---------------------------------------------------------------------------------------------
// PatchMatch helpers

function pmAgreement(g: DepthMap, c: DepthMap, gt: Float32Array): { agreement: number; gpuValid: number; cpuValid: number } {
  let both = 0;
  let agree = 0;
  let gv = 0;
  let cv = 0;
  let n = 0;
  for (let i = 0; i < g.depth.length; i++) {
    if (!(gt[i] > 0)) continue;
    n++;
    const a = g.depth[i];
    const b = c.depth[i];
    if (a > 0) gv++;
    if (b > 0) cv++;
    if (a > 0 || b > 0) {
      both++;
      if (a > 0 && b > 0 && Math.abs(a - b) <= 0.02 * b) agree++;
    }
  }
  return { agreement: both ? agree / both : 0, gpuValid: n ? gv / n : 0, cpuValid: n ? cv / n : 0 };
}

function pmAccuracy(dm: DepthMap, ds: SynthDataset): { valid: number; acc1: number; acc2: number; medRel: number } {
  const mask = surfaceMask(ds.views[0].cam, ds.gtDepth[0], ds.gtNormal[0]);
  const s1 = evaluateDepthMap(dm, ds.gtDepth[0], ds.gtNormal[0], { mask, relTol: 0.01 });
  const s2 = evaluateDepthMap(dm, ds.gtDepth[0], ds.gtNormal[0], { mask, relTol: 0.02 });
  return { valid: s1.validFraction, acc1: s1.accurateFraction, acc2: s2.accurateFraction, medRel: s1.medianRelError };
}

// ---------------------------------------------------------------------------------------------

export async function runBenchmark(o: RunOptions): Promise<BenchReport> {
  const cfg = SCALES[o.scale];
  const t0 = now();
  const startedAt = new Date().toISOString();
  const nav = (globalThis as { navigator?: Navigator }).navigator;
  const device: BenchDevice = {
    userAgent: nav?.userAgent ?? 'unknown',
    hardwareConcurrency: typeof nav?.hardwareConcurrency === 'number' ? nav.hardwareConcurrency : null,
    gpu: { available: false },
  };
  const rates: BenchRates = {
    siftGpuMsPerMP: null,
    siftCpuMsPerMP: null,
    matchGpuMsPerMPairs: null,
    matchCpuMsPerMPairs: null,
    pmGpuMsPerUnit: null,
    pmCpuMsPerUnit: null,
    baMsPerIterPerKObs: null,
  };
  const tests: BenchTestResult[] = [];
  const emit = (r: BenchTestResult): void => {
    try {
      o.onTest?.(JSON.parse(JSON.stringify(r)) as BenchTestResult);
    } catch {
      /* listener problems never stop the run */
    }
  };
  const progress = (id: BenchTestId, m: string): void => {
    try {
      o.onProgress?.(id, m);
    } catch {
      /* ignore */
    }
  };
  const alive = (): void => {
    if (o.cancel.cancelled) throw new Cancelled();
  };

  // ---- GPU context ------------------------------------------------------------------------------
  let ctx: GPUContext | null = null;
  if (!o.gpu) device.gpu.reason = o.noGpuReason ?? 'WebGPU disabled for this run';
  else if (!(nav as Navigator & { gpu?: unknown } | undefined)?.gpu) device.gpu.reason = 'WebGPU is not available (older iOS, Lockdown Mode or browser setting)';
  else {
    try {
      const { getGPU } = await import('../../gpu/device');
      ctx = await getGPU();
      if (!ctx) device.gpu.reason = 'No WebGPU adapter';
    } catch (e) {
      device.gpu.reason = `WebGPU initialisation failed: ${errMsg(e)}`;
    }
  }
  if (ctx) {
    const info = ctx.info;
    device.gpu = {
      available: true,
      adapter: [info.vendor, info.architecture, info.description].filter(Boolean).join(' · ') || 'unknown adapter',
      isSoftware: ctx.isSoftware,
      maxBufferSize: ctx.limits.maxBufferSize,
      maxStorageBufferBindingSize: ctx.limits.maxStorageBufferBindingSize,
    };
  }
  const noGpuNote = `${device.gpu.reason ?? 'No WebGPU'} — CPU fallback measured.`;

  const pending = BENCH_TESTS.map((t) => newResult(t.id, !!ctx));
  for (const p of pending) {
    p.status = 'pending';
    emit(p);
  }
  let cancelled = false;

  const runTest = async (id: BenchTestId, body: (r: BenchTestResult) => Promise<void>): Promise<void> => {
    const r = newResult(id, !!ctx);
    if (cancelled) {
      r.status = 'cancelled';
      tests.push(r);
      emit(r);
      return;
    }
    if (!ctx && id !== 'ba') r.note = noGpuNote;
    emit(r);
    const ts = now();
    try {
      await yieldTick();
      alive();
      await body(r);
    } catch (e) {
      if (e instanceof Cancelled || o.cancel.cancelled) {
        cancelled = true;
        r.status = 'cancelled';
      } else {
        r.error = errMsg(e);
        r.status = 'error';
      }
    }
    finish(r, ts);
    tests.push(r);
    emit(r);
  };

  // ---- 1. SIFT ----------------------------------------------------------------------------------
  await runTest('sift', async (r) => {
    const [pw, ph] = cfg.parity;
    progress('sift', `Rendering ${pw}×${ph} test image…`);
    const small = makeTexturedImage(pw, ph, 5);
    await yieldTick();
    alive();
    progress('sift', `CPU SIFT ${pw}×${ph}…`);
    extractSiftCPU(small, SIFT_OPTS); // warm-up (JIT, scratch arena)
    alive();
    const tc = now();
    const cpu = extractSiftCPU(small, SIFT_OPTS);
    const cpuMs = now() - tc;
    const cpuStages = { ...lastSiftCPUTimings };
    rates.siftCpuMsPerMP = cpuMs / ((pw * ph) / 1e6);
    r.timings[`CPU ${pw}×${ph}`] = r1(cpuMs);
    r.metrics.cpuFeatures = cpu.count;
    if (typeof cpuStages.simd === 'number') r.metrics.cpuSimd = cpuStages.simd;
    r.checks.push(check('CPU features found', cpu.count, '>=', Math.min(50, (pw * ph) / 2000), (v) => String(Math.round(v))));
    if (!ctx) {
      r.summary = `CPU ${pw}×${ph}: ${Math.round(cpuMs)} ms · ${cpu.count} features (${Math.round(rates.siftCpuMsPerMP)} ms/MP)`;
      releaseSiftCPUMemory();
      return;
    }
    await yieldTick();
    alive();
    const { SiftGPU } = await import('../../features/sift-gpu');
    progress('sift', 'Compiling GPU pipelines…');
    const tcomp = now();
    const sift = await SiftGPU.create(ctx, SIFT_OPTS);
    r.timings['GPU pipeline compile'] = r1(now() - tcomp);
    try {
      alive();
      progress('sift', `GPU vs CPU parity ${pw}×${ph}…`);
      const g = await sift.extract(small);
      const par = featureParity(cpu, g);
      r.metrics.parityGpuFeatures = g.count;
      r.metrics.parityPos = r3(par.posFrac);
      r.metrics.parityMedianCos = Math.round(par.medianCos * 1e4) / 1e4;
      r.checks.push(check('GPU/CPU feature count', Math.abs(g.count - cpu.count) / Math.max(1, cpu.count), '<=', 0.05, pct));
      r.checks.push(check('GPU/CPU keypoints within 1 px', par.posFrac, '>=', 0.9, pct));
      r.checks.push(check('GPU/CPU descriptor cosine (median)', par.medianCos, '>=', 0.97, (v) => v.toFixed(4)));
      const [W, H] = cfg.sift;
      progress('sift', `Rendering ${W}×${H} test image…`);
      const big = makeTexturedImage(W, H, 9);
      const id = grayToImageData(big);
      let bitmap: ImageBitmap | null = null;
      if (id && typeof createImageBitmap === 'function') {
        try {
          bitmap = await createImageBitmap(id);
        } catch {
          bitmap = null;
        }
      }
      const extract = (): Promise<FeatureSet> => (bitmap ? sift.extractFromBitmap(bitmap) : sift.extract(big));
      alive();
      progress('sift', `GPU ${W}×${H} warm-up…`);
      const tw = now();
      await extract();
      r.timings['GPU first run (allocation)'] = r1(now() - tw);
      const runs: { ms: number; t: NonNullable<typeof sift.lastTimings>; count: number }[] = [];
      for (let i = 0; i < cfg.siftRuns; i++) {
        alive();
        progress('sift', `GPU ${W}×${H} run ${i + 1}/${cfg.siftRuns}…`);
        const ts = now();
        const fs = await extract();
        runs.push({ ms: now() - ts, t: sift.lastTimings!, count: fs.count });
      }
      bitmap?.close();
      runs.sort((x, y) => x.ms - y.ms);
      const med = runs[runs.length >> 1];
      const T = med.t;
      r.timings[`GPU ${W}×${H} total`] = r1(med.ms);
      r.timings['upload'] = r1(T.uploadMs);
      r.timings['detect (pyramid + DoG)'] = r1(T.detectMs);
      r.timings['select'] = r1(T.selectMs);
      r.timings['describe (orientation + descriptor)'] = r1(T.describeMs);
      r.timings['assemble'] = r1(T.assembleMs);
      r.metrics.gpuFeatures = med.count;
      r.metrics.candidates = T.candidates;
      r.metrics.submits = T.submits;
      r.metrics.readbacks = T.readbacks;
      r.metrics.input = bitmap ? 1 : 0;
      rates.siftGpuMsPerMP = med.ms / ((W * H) / 1e6);
      r.checks.push(check(`GPU features ${W}×${H}`, med.count, '>=', Math.min(1000, (W * H) / 400), (v) => String(Math.round(v))));
      r.summary = `GPU ${W}×${H}: ${Math.round(med.ms)} ms · ${med.count} features · parity ${pct(par.posFrac)} / cos ${par.medianCos.toFixed(3)} · CPU ${pw}×${ph} ${Math.round(cpuMs)} ms`;
    } finally {
      sift.destroy();
      releaseSiftCPUMemory();
    }
  });

  // ---- 2. matcher -------------------------------------------------------------------------------
  await runTest('matcher', async (r) => {
    const n = cfg.match;
    progress('matcher', `Generating ${n} × ${n} descriptors…`);
    const { a, b, truth } = makeMatchSets(n);
    await yieldTick();
    alive();
    progress('matcher', `CPU matching ${n} × ${n}…`);
    const tc = now();
    const mc = matchCPU(a, b);
    const cpuMs = now() - tc;
    const mpairs = (n * n) / 1e6;
    rates.matchCpuMsPerMPairs = cpuMs / mpairs;
    r.timings[`CPU ${n}×${n}`] = r1(cpuMs);
    const rc = matchRecall(mc, truth);
    r.metrics.cpuMatches = mc.length / 2;
    r.metrics.recall = r3(rc.recall);
    r.metrics.precision = r3(rc.precision);
    r.checks.push(check('recall of the true correspondences', rc.recall, '>=', 0.8, pct));
    r.checks.push(check('precision', rc.precision, '>=', 0.98, pct));
    if (!ctx) {
      r.summary = `CPU ${n}×${n}: ${Math.round(cpuMs)} ms · ${mc.length / 2} matches · recall ${pct(rc.recall)}`;
      return;
    }
    alive();
    const { MatcherGPU } = await import('../../features/match-gpu');
    progress('matcher', 'Compiling GPU pipelines…');
    const tcomp = now();
    const m = await MatcherGPU.create(ctx);
    r.timings['GPU pipeline compile'] = r1(now() - tcomp);
    try {
      alive();
      progress('matcher', `GPU matching ${n} × ${n}…`);
      const t1 = now();
      const g1 = await m.match(a, b);
      const firstMs = now() - t1;
      alive();
      const t2 = now();
      const g2 = await m.match(a, b);
      const cachedMs = now() - t2;
      const T = m.lastTimings;
      r.timings[`GPU ${n}×${n} first (incl. upload)`] = r1(firstMs);
      r.timings[`GPU ${n}×${n} cached`] = r1(cachedMs);
      if (T) {
        r.timings['prepare'] = r1(T.prepareMs);
        r.timings['GPU + readback'] = r1(T.gpuMs);
        r.timings['finalize'] = r1(T.finalizeMs);
      }
      rates.matchGpuMsPerMPairs = cachedMs / mpairs;
      const ov = Math.min(pairOverlap(mc, g1), pairOverlap(mc, g2));
      r.metrics.gpuMatches = g2.length / 2;
      r.metrics.parityOverlap = r3(ov);
      r.checks.push(check('GPU/CPU identical pairs', ov, '>=', 0.99, pct));
      r.summary = `GPU ${n}×${n}: ${Math.round(cachedMs)} ms (first ${Math.round(firstMs)} ms) · CPU ${Math.round(cpuMs)} ms · ${g2.length / 2} matches, ${pct(ov)} identical`;
    } finally {
      m.destroy();
    }
  });

  // ---- 3. PatchMatch ----------------------------------------------------------------------------
  await runTest('patchmatch', async (r) => {
    const pc = cfg.pmParity;
    progress('patchmatch', `Rendering synthetic scene ${pc.w}×${pc.h}, ${pc.sources + 1} views…`);
    const tr = now();
    const dsP = makeSyntheticDataset('near', { width: pc.w, height: pc.h, numSources: pc.sources, supersample: 1 });
    r.timings['render scene (CPU)'] = r1(now() - tr);
    await yieldTick();
    alive();
    const pmOpts = { iterations: pc.iterations };
    progress('patchmatch', `CPU PatchMatch ${pc.w}×${pc.h}…`);
    const tc = now();
    const dc = patchMatchCPU(dsP.views[0], dsP.views.slice(1), dsP.range, pmOpts);
    const cpuMs = now() - tc;
    const cpuUnits = ((pc.w * pc.h) / 1e6) * pc.sources * pc.iterations;
    rates.pmCpuMsPerUnit = cpuMs / cpuUnits;
    r.timings[`CPU ${pc.w}×${pc.h} · ${pc.sources} src · ${pc.iterations} it`] = r1(cpuMs);
    const accC = pmAccuracy(dc, dsP);
    r.metrics.cpuValid = r3(accC.valid);
    r.metrics.cpuAccurate2pct = r3(accC.acc2);
    if (!ctx) {
      r.checks.push(check('CPU valid pixels (textured surfaces)', accC.valid, '>=', 0.6, pct));
      r.checks.push(check('CPU depth within 2 % of ground truth', accC.acc2, '>=', 0.75, pct));
      r.summary = `CPU ${pc.w}×${pc.h}, ${pc.sources} sources: ${Math.round(cpuMs)} ms · ${pct(accC.valid)} valid, ${pct(accC.acc2)} within 2 %`;
      return;
    }
    alive();
    const { PatchMatchGPU } = await import('../../dense/patchmatch-gpu');
    progress('patchmatch', 'Compiling GPU pipelines…');
    const tcomp = now();
    const pm = await PatchMatchGPU.create(ctx);
    r.timings['GPU pipeline compile'] = r1(now() - tcomp);
    try {
      alive();
      progress('patchmatch', `GPU vs CPU ${pc.w}×${pc.h}…`);
      const gP = await pm.compute(dsP.views[0], dsP.views.slice(1), dsP.range, pmOpts);
      const ag = pmAgreement(gP, dc, dsP.gtDepth[0]);
      r.metrics.parityAgreement = r3(ag.agreement);
      r.checks.push(check('GPU/CPU depth agreement (2 %)', ag.agreement, '>=', 0.7, pct));
      r.checks.push(check('GPU/CPU valid-pixel difference', Math.abs(ag.gpuValid - ag.cpuValid), '<', 0.1, pct));
      const c = cfg.pm;
      let ds = dsP;
      if (c.w !== pc.w || c.h !== pc.h || c.sources !== pc.sources) {
        alive();
        progress('patchmatch', `Rendering synthetic scene ${c.w}×${c.h}, ${c.sources + 1} views…`);
        const tr2 = now();
        ds = makeSyntheticDataset('near', { width: c.w, height: c.h, numSources: c.sources, supersample: 1 });
        r.timings[`render ${c.w}×${c.h} (CPU)`] = r1(now() - tr2);
        await yieldTick();
      }
      alive();
      progress('patchmatch', `GPU ${c.w}×${c.h}, ${c.sources} sources, ${c.iterations} iterations…`);
      const tg = now();
      const dm = await pm.compute(ds.views[0], ds.views.slice(1), ds.range, { iterations: c.iterations });
      const gpuMs = now() - tg;
      const T = pm.lastTimings!;
      const perIt = T.iterationsMs.length ? T.iterationsMs.reduce((x, y) => x + y, 0) / T.iterationsMs.length : NaN;
      r.timings[`GPU ${c.w}×${c.h} · ${c.sources} src total`] = r1(gpuMs);
      r.timings['upload'] = r1(T.uploadMs);
      r.timings['init'] = r1(T.initMs);
      T.iterationsMs.forEach((ms, i) => (r.timings[`iteration ${i + 1}`] = r1(ms)));
      r.timings['readback'] = r1(T.readbackMs);
      r.metrics.maxSubmitMs = r1(T.maxSubmitMs);
      r.metrics.submits = T.submits;
      rates.pmGpuMsPerUnit = perIt / (((c.w * c.h) / 1e6) * c.sources);
      const acc = pmAccuracy(dm, ds);
      r.metrics.valid = r3(acc.valid);
      r.metrics.accurate1pct = r3(acc.acc1);
      r.metrics.accurate2pct = r3(acc.acc2);
      r.metrics.medianRelError = Math.round(acc.medRel * 1e5) / 1e5;
      const full = c.w >= 320;
      r.checks.push(check('valid pixels (textured surfaces)', acc.valid, '>=', full ? 0.7 : 0.6, pct));
      r.checks.push(check(full ? 'depth within 1 % of ground truth' : 'depth within 2 % of ground truth', full ? acc.acc1 : acc.acc2, '>=', full ? 0.8 : 0.75, pct));
      r.summary = `GPU ${c.w}×${c.h}, ${c.sources} sources: ${Math.round(perIt)} ms/iteration · ${pct(acc.valid)} valid, ${pct(full ? acc.acc1 : acc.acc2)} within ${full ? 1 : 2} % · CPU/GPU agree ${pct(ag.agreement)}`;
    } finally {
      pm.destroy();
    }
  });

  // ---- 4. bundle adjustment ---------------------------------------------------------------------
  await runTest('ba', async (r) => {
    const c = cfg.ba;
    progress('ba', `Generating ${c.poses}-image problem…`);
    const s = makeSyntheticBA({ numPoses: c.poses, numPoints: c.points, layout: 'strafe', trackWindow: 2, radius: 0.3, k1: -0.05, noisePx: 0.5, seed: 3 });
    const p = s.problem;
    fixGaugeTwoPoses(p);
    p.cameraRefine[0] = { focal: true, principalPoint: false, distortion: true };
    perturbBA(s, { rotSigma: 0.002, centreSigma: 0.02, pointSigma: 0.1, focalScale: 0.02, k1Delta: 0.01 });
    const nObs = p.obsPose.length;
    await yieldTick();
    alive();
    progress('ba', `LM iterations on ${c.poses} images, ${c.points} points, ${nObs} observations…`);
    const times: number[] = [];
    let last = now();
    const res = bundleAdjust(p, {
      maxIterations: c.iterations,
      functionTolerance: 0,
      cancel: o.cancel,
      onIteration: () => {
        const t = now();
        times.push(t - last);
        last = t;
      },
    });
    alive();
    const steady = times.length > 1 ? times.slice(1) : times;
    const perIt = median(steady);
    rates.baMsPerIterPerKObs = perIt / (nObs / 1000);
    times.forEach((ms, i) => (r.timings[i === 0 ? 'iteration 1 (incl. setup)' : `iteration ${i + 1}`] = r1(ms)));
    r.timings.total = r1(res.timeMs);
    const e = syntheticErrors(s);
    r.metrics.images = c.poses;
    r.metrics.points = c.points;
    r.metrics.observations = nObs;
    r.metrics.iterations = res.iterations;
    r.metrics.initialRmsPx = r3(res.initialRmsPx);
    r.metrics.finalRmsPx = r3(res.finalRmsPx);
    r.metrics.focalErrorPct = r3(100 * e.maxFocalRelErr);
    r.metrics.reducedParams = res.numReducedParams;
    r.checks.push({ name: 'solver succeeded', pass: res.success, detail: `${res.termination} (${res.linearSolver})` });
    r.checks.push(check('final reprojection RMS', res.finalRmsPx, '<', 0.8, (v) => `${v.toFixed(3)} px`));
    r.summary = `${c.poses} images, ${nObs} observations: ${Math.round(perIt)} ms/iteration (${res.linearSolver}) · RMS ${res.initialRmsPx.toFixed(1)} → ${res.finalRmsPx.toFixed(2)} px`;
  });

  const keyframes = o.keyframes ?? 80;
  const estimates = cancelled ? [] : estimateScanTimes(rates, { keyframes });
  return {
    version: 1,
    startedAt,
    durationMs: Math.round(now() - t0),
    scale: o.scale,
    cancelled,
    device,
    tests,
    rates: Object.fromEntries(Object.entries(rates).map(([k, v]) => [k, v === null ? null : Math.round(v * 1000) / 1000])) as unknown as BenchRates,
    estimates,
    assumptions: estimateAssumptions(keyframes),
  };
}
