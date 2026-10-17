/**
 * Multiplier for wall-clock assertions in performance tests. Shared CI runners
 * and loaded dev machines are slower than the reference machine, so CI sets
 * PERF_SLACK (e.g. 4) to relax time limits without disabling the tests.
 */
export function perfSlack(): number {
  const env = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env;
  const v = Number(env?.PERF_SLACK);
  return Number.isFinite(v) && v >= 1 ? v : 1;
}
