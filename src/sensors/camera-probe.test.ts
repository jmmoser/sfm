import { describe, expect, it } from 'vitest';
import { analyzeCameraProbe, dist, type ProbeMstpSample, type ProbeVideoSample } from './camera-probe';

/** 30 fps for `n` frames: capture clock starts at 50 s; the page timeline is offset by −49 900 ms. */
function synth(n: number, opts: { jitter?: (i: number) => number; driftMsPerS?: number; theta?: (i: number) => number; portraitAt?: number } = {}) {
  const video: ProbeVideoSample[] = [];
  const mstp: ProbeMstpSample[] = [];
  for (let i = 0; i < n; i++) {
    const tsMs = 50_000 + (i * 1000) / 30;
    const page = tsMs - 49_900; // capture time on the performance.now() timeline
    const arrival = page + 4 + (i % 3); // worker arrival: 4–6 ms after capture
    const capture = page + 20 + (opts.jitter?.(i) ?? 0) + ((opts.driftMsPerS ?? 0) * (i / 30)); // rVFC captureTime: post-ISP
    const portrait = opts.portraitAt !== undefined && i >= opts.portraitAt;
    mstp.push({ timestampUs: tsMs * 1000, receivedMs: arrival, codedWidth: portrait ? 1440 : 1920, codedHeight: portrait ? 1920 : 1440, displayWidth: 1920, displayHeight: 1440, format: 'NV12' });
    const theta = opts.theta?.(i) ?? 0;
    video.push({ captureTime: capture, presentationTime: capture + 10, callbackTime: capture + 12, width: theta % 180 ? 1920 : 1440, height: theta % 180 ? 1440 : 1920, screenAngle: theta });
  }
  return { video, mstp };
}

describe('analyzeCameraProbe', () => {
  it('a constant rVFC lag shows as an offset, not as jitter or drift', () => {
    const { video, mstp } = synth(300);
    const r = analyzeCameraProbe({ durationMs: 10_000, video, mstp, grabs: [{ syncMs: 0.4, totalMs: 6 }], mstpError: null, mstpTransport: 'track' });
    expect(r.timing.pairs).toBe(300);
    expect(r.timing.offsetMedianMs).toBeCloseTo(20 - 49_900, 6);
    expect(r.timing.jitterStdMs).toBe(0);
    expect(r.timing.driftMsPerS).toBe(0);
    expect(r.video).toMatchObject({ frames: 300, fps: 30, timeSource: 'captureTime' });
    expect(r.video.intervalMs!.p50).toBeCloseTo(1000 / 30, 6);
    expect(r.mstp).toMatchObject({ frames: 300, alwaysLandscape: true, codedSizes: ['1920x1440'], formats: ['NV12'], transport: 'track' });
    expect(r.mstp.arrivalLatencyJitterMs!.max).toBe(2);
    expect(r.summary.join('\n')).toMatch(/small: MSTP timestamps would gain little/);
    expect(r.summary.join('\n')).toMatch(/Rotate the phone/);
  });

  it('measures rVFC jitter and drift against the capture clock', () => {
    // ±3 ms alternating jitter and 1 ms/s drift
    const { video, mstp } = synth(300, { jitter: (i) => (i % 2 ? 3 : -3), driftMsPerS: 1 });
    const r = analyzeCameraProbe({ durationMs: 10_000, video, mstp, grabs: [], mstpError: null, mstpTransport: 'track' });
    expect(r.timing.pairs).toBe(300);
    expect(r.timing.driftMsPerS!).toBeCloseTo(1, 1);
    expect(r.timing.jitterStdMs!).toBeCloseTo(3, 1); // the injected ±3 ms; drift is not counted again
    expect(r.timing.jitterRangeMs!).toBeCloseTo(6, 1);
    expect(r.summary.join('\n')).toMatch(/material: MSTP timestamps would help/);
    expect(r.grab.syncMs).toBeNull();
  });

  it('does not mispair around a frame MSTP dropped, and records orientation per UI rotation', () => {
    const { video, mstp } = synth(90, { theta: (i) => (i < 45 ? 0 : 90), portraitAt: 80 });
    mstp.splice(40, 1); // dropped by the processor queue
    const r = analyzeCameraProbe({ durationMs: 3000, video, mstp, grabs: [], mstpError: null, mstpTransport: 'stream' });
    expect(r.timing.pairs).toBe(89); // rVFC frame 40 has no partner: not paired with a neighbour
    expect(r.timing.jitterStdMs).toBe(0);
    expect(r.video.sizesByTheta).toEqual({ '0': ['1440x1920'], '90': ['1920x1440'] });
    expect(r.mstp.alwaysLandscape).toBe(false);
    expect(r.summary.join('\n')).toMatch(/NOT always landscape/);
  });

  it('reports a missing MSTP path and rVFC without captureTime', () => {
    const { video } = synth(30);
    for (const v of video) v.captureTime = null;
    const r = analyzeCameraProbe({ durationMs: 1000, video, mstp: [], grabs: [], mstpError: 'MediaStreamTrackProcessor is not available in workers', mstpTransport: null });
    expect(r.video.timeSource).toBe('presentationTime');
    expect(r.mstp).toMatchObject({ available: false, frames: 0, alwaysLandscape: null });
    expect(r.timing).toMatchObject({ pairs: 0, jitterStdMs: null, driftMsPerS: null });
    expect(r.summary[0]).toMatch(/unavailable: MediaStreamTrackProcessor is not available/);
  });

  it('dist', () => {
    expect(dist([])).toBeNull();
    expect(dist([3, 1, 2, NaN])).toEqual({ n: 3, p50: 2, p95: 3, max: 3 });
  });
});
