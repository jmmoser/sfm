/**
 * Text form of a benchmark report ("Copy report": this text followed by the JSON) and small
 * helpers shared by the Diagnostics screen. Pure (no DOM): unit-tested in report.test.ts.
 */
import { formatBytes, formatDuration } from '../format';
import type { BenchEstimate, BenchReport, BenchStatus, BenchTestResult } from './types';

export const STATUS_LABEL: Record<BenchStatus, string> = {
  pending: 'Pending',
  running: 'Running',
  pass: 'Pass',
  fail: 'Fail',
  skipped: 'Skipped',
  cancelled: 'Cancelled',
  error: 'Error',
};

/** Milliseconds for timing lists: 0.8 ms, 42 ms, 1.24 s, 2 min 5 s. */
export function formatMs(ms: number): string {
  if (!Number.isFinite(ms)) return '–';
  if (ms < 10) return `${ms.toFixed(1)} ms`;
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(2)} s`;
  return formatDuration(ms);
}

/** "3 passed · 1 failed · 1 skipped". */
export function verdict(tests: readonly BenchTestResult[]): { pass: number; fail: number; other: number; text: string } {
  let pass = 0;
  let fail = 0;
  let other = 0;
  for (const t of tests) {
    if (t.status === 'pass') pass++;
    else if (t.status === 'fail' || t.status === 'error') fail++;
    else other++;
  }
  const parts = [`${pass} passed`];
  if (fail) parts.push(`${fail} failed`);
  if (other) parts.push(`${other} not run`);
  return { pass, fail, other, text: parts.join(' · ') };
}

export function estimateLine(e: BenchEstimate): string {
  const fps = e.captureFps >= 10 ? '10 fps (full rate)' : e.captureFps < 0.1 ? '< 0.1 fps' : `${e.captureFps.toFixed(1)} fps`;
  return `capture ${fps} · refine ${formatDuration(e.refineMs)} · dense ${formatDuration(e.denseMs)} · mesh ${formatDuration(e.meshMs)}`;
}

const QUALITY: Record<BenchEstimate['quality'], string> = { fast: 'Fast', balanced: 'Balanced', max: 'Max' };

export function qualityName(q: BenchEstimate['quality']): string {
  return QUALITY[q];
}

/** Human-readable report (plain text, for bug reports / chat). */
export function formatReportText(r: BenchReport): string {
  const L: string[] = [];
  L.push('3D Scan — benchmark & self-test');
  L.push(`Scale: ${r.scale} · ${r.startedAt} · ${formatDuration(r.durationMs)}${r.cancelled ? ' · CANCELLED' : ''}`);
  if (r.error) L.push(`Run ended early: ${r.error}`);
  L.push(`Device: ${r.device.userAgent}${r.device.hardwareConcurrency ? ` · ${r.device.hardwareConcurrency} cores` : ''}`);
  const g = r.device.gpu;
  L.push(
    g.available
      ? `WebGPU: ${g.adapter ?? 'adapter'}${g.isSoftware ? ' (software renderer)' : ''}${g.maxBufferSize ? ` · maxBufferSize ${formatBytes(g.maxBufferSize)}` : ''}`
      : `WebGPU: unavailable — ${g.reason ?? 'unknown reason'}`,
  );
  L.push(`Result: ${verdict(r.tests).text}`);
  for (const t of r.tests) {
    L.push('');
    L.push(`[${STATUS_LABEL[t.status].toUpperCase()}] ${t.title} (${t.backend.toUpperCase()}) — ${t.summary || t.error || t.note || ''}`.trimEnd());
    if (t.note && t.summary) L.push(`  note: ${t.note}`);
    if (t.error) L.push(`  error: ${t.error}`);
    const tm = Object.entries(t.timings);
    if (tm.length) L.push(`  timings: ${tm.map(([k, v]) => `${k} ${formatMs(v)}`).join(' · ')}`);
    for (const c of t.checks) L.push(`  ${c.pass ? '✓' : '✗'} ${c.name}: ${c.detail}`);
  }
  if (r.estimates.length) {
    L.push('');
    const e0 = r.estimates[0];
    L.push(`Estimated processing of a scan with ${e0.keyframes} keyframes (rough; features ${e0.features.toUpperCase()}, dense ${e0.dense.toUpperCase()}):`);
    for (const e of r.estimates) L.push(`  ${qualityName(e.quality).padEnd(9)} ${estimateLine(e)} → ${formatDuration(e.processingMs)} after capture`);
    L.push(`Assumptions: ${r.assumptions.join('; ')}.`);
  }
  return L.join('\n');
}

/** Text + JSON (what "Copy report" puts on the clipboard). */
export function reportClipboardText(r: BenchReport): string {
  return `${formatReportText(r)}\n\n--- JSON ---\n${JSON.stringify(r, null, 2)}\n`;
}
