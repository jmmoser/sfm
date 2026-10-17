/**
 * Main-thread side of the on-device benchmark: runs runner.ts in a dedicated module worker
 * (bench.worker.ts) so the UI stays responsive, relays per-test results / progress, and cancels.
 *
 * Cancel posts `cancel` (checked by the runner between steps and per BA iteration); a synchronous
 * CPU step cannot see it, so the worker is TERMINATED when it has not finished within
 * `cancelGraceMs` and the report is completed from the results received so far. The worker is
 * always terminated at the end (its GPU device and buffers go with it). A crashed worker (module
 * load failure, uncaught error) yields a report with `error` instead of a rejected promise.
 */
import { BENCH_TESTS, type BenchFromWorker, type BenchReport, type BenchScale, type BenchTestId, type BenchTestResult, type BenchToWorker } from './types';

export interface BenchRunOptions {
  scale: BenchScale;
  /** Use WebGPU (false: CPU fallbacks only). */
  gpu: boolean;
  /** Why `gpu` is false (report). */
  noGpuReason?: string;
  onTest?: (r: BenchTestResult) => void;
  onProgress?: (id: BenchTestId, message: string) => void;
  /** After a cancel, terminate the worker when it has not stopped within this long (default 2.5 s). */
  cancelGraceMs?: number;
  /** Tests inject a worker. */
  createWorker?: () => Worker;
}

export interface BenchRun {
  /** Resolves with the final report (never rejects). */
  readonly done: Promise<BenchReport>;
  cancel(reason?: string): void;
}

function defaultWorker(): Worker {
  return new Worker(new URL('./bench.worker.ts', import.meta.url), { type: 'module', name: 'bench' });
}

export function startBenchmark(o: BenchRunOptions): BenchRun {
  const startedAt = new Date().toISOString();
  const t0 = performance.now();
  const latest = new Map<BenchTestId, BenchTestResult>();
  let worker: Worker | null = null;
  let settled = false;
  let resolveDone!: (r: BenchReport) => void;
  const done = new Promise<BenchReport>((r) => (resolveDone = r));
  let cancelTimer: ReturnType<typeof setTimeout> | null = null;
  let cancelReason: string | undefined;

  const finish = (r: BenchReport): void => {
    if (settled) return;
    settled = true;
    if (cancelTimer) clearTimeout(cancelTimer);
    try {
      worker?.terminate();
    } catch {
      /* ignore */
    }
    worker = null;
    resolveDone(r);
  };

  /** Report from what arrived so far (worker terminated / crashed). */
  const partial = (cancelled: boolean, error?: string): BenchReport => {
    const tests = BENCH_TESTS.map(({ id, title }) => {
      const t = latest.get(id);
      const base: BenchTestResult = t ? { ...t } : { id, title, status: 'pending', backend: o.gpu ? 'gpu' : 'cpu', summary: '', timings: {}, metrics: {}, checks: [], ms: 0 };
      if (base.status === 'running' || base.status === 'pending') {
        base.status = cancelled ? 'cancelled' : 'error';
        if (!cancelled && error) base.error = error;
      }
      return base;
    });
    const nav = typeof navigator !== 'undefined' ? navigator : undefined;
    return {
      version: 1,
      startedAt,
      durationMs: Math.round(performance.now() - t0),
      scale: o.scale,
      cancelled,
      device: { userAgent: nav?.userAgent ?? 'unknown', hardwareConcurrency: nav?.hardwareConcurrency ?? null, gpu: { available: false, reason: 'not reported (run ended early)' } },
      tests,
      rates: { siftGpuMsPerMP: null, siftCpuMsPerMP: null, matchGpuMsPerMPairs: null, matchCpuMsPerMPairs: null, pmGpuMsPerUnit: null, pmCpuMsPerUnit: null, baMsPerIterPerKObs: null },
      estimates: [],
      assumptions: [],
      ...(error ? { error } : {}),
    };
  };

  try {
    worker = (o.createWorker ?? defaultWorker)();
  } catch (e) {
    finish(partial(false, `The benchmark worker could not start: ${(e as Error)?.message ?? e}`));
    return { done, cancel: () => undefined };
  }
  worker.addEventListener('message', (ev: MessageEvent<BenchFromWorker>) => {
    const m = ev.data;
    if (!m || settled) return;
    switch (m.type) {
      case 'test':
        latest.set(m.result.id, m.result);
        o.onTest?.(m.result);
        break;
      case 'progress':
        o.onProgress?.(m.id, m.message);
        break;
      case 'done':
        finish(m.report.cancelled && cancelReason && !m.report.error ? { ...m.report, error: cancelReason } : m.report);
        break;
      case 'fatal':
        finish(partial(false, m.message));
        break;
    }
  });
  worker.addEventListener('error', (ev: ErrorEvent) => {
    ev.preventDefault?.();
    finish(partial(false, `The benchmark worker crashed: ${ev.message || 'unknown error'}`));
  });
  const run: BenchToWorker = { type: 'run', scale: o.scale, gpu: o.gpu, ...(o.noGpuReason ? { noGpuReason: o.noGpuReason } : {}) };
  worker.postMessage(run);

  return {
    done,
    cancel(reason?: string): void {
      if (settled || !worker) return;
      cancelReason ??= reason;
      const msg: BenchToWorker = { type: 'cancel' };
      worker.postMessage(msg);
      if (cancelTimer) return;
      cancelTimer = setTimeout(() => finish(partial(true, reason)), o.cancelGraceMs ?? 2500);
    },
  };
}
