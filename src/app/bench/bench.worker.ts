/**
 * Benchmark worker (module worker): runs runner.ts off the UI thread, so the Diagnostics screen
 * stays responsive while the GPU / CPU are busy. One run per worker; the UI terminates the worker
 * afterwards (releasing its GPU device), or when a cancel does not take effect in time.
 */
import { runBenchmark } from './runner';
import type { BenchFromWorker, BenchToWorker } from './types';

const scope = self as unknown as {
  onmessage: ((ev: MessageEvent<unknown>) => void) | null;
  postMessage(msg: BenchFromWorker): void;
  addEventListener(type: string, cb: (ev: Event) => void): void;
};

let token: { cancelled: boolean } | null = null;

function post(msg: BenchFromWorker): void {
  try {
    scope.postMessage(msg);
  } catch (e) {
    console.error('[bench.worker] postMessage failed', e);
  }
}

scope.onmessage = (ev: MessageEvent<unknown>): void => {
  const m = ev.data as BenchToWorker | null;
  if (!m || typeof m !== 'object') return;
  if (m.type === 'cancel') {
    if (token) token.cancelled = true;
    return;
  }
  if (m.type !== 'run' || token) return;
  const t = { cancelled: false };
  token = t;
  runBenchmark({
    scale: m.scale,
    gpu: m.gpu,
    noGpuReason: m.noGpuReason,
    cancel: t,
    onTest: (result) => post({ type: 'test', result }),
    onProgress: (id, message) => post({ type: 'progress', id, message }),
  }).then(
    (report) => post({ type: 'done', report }),
    (e: unknown) => post({ type: 'fatal', message: (e as Error)?.message ?? String(e) }),
  );
};

scope.addEventListener('unhandledrejection', (ev: Event) => {
  const reason = (ev as PromiseRejectionEvent).reason as Error | undefined;
  console.warn('[bench.worker] unhandled rejection', reason?.message ?? reason);
});
