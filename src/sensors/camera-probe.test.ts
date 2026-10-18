import { describe, expect, it } from 'vitest';
import { analyzeCameraProbe, dist, type ProbeGrabSample, type ProbeMstpSample, type ProbeVideoSample } from './camera-probe';

interface SynthOpts {
  /** Extra rVFC lag of frame i (ms). */
  jitter?: (i: number) => number;
  driftMsPerS?: number;
  /** UI rotation of frame i. */
  theta?: (i: number) => number;
  /** rVFC frame i has the wrong orientation for its θ. */
  misoriented?: (i: number) => boolean;
  /** rVFC frame i is missing (dropped). */
  dropVideo?: (i: number) => boolean;
  portraitMstpAt?: number;
}

/** 30 fps for `n` frames: capture clock starts at 50 s; the page timeline is offset by −49 900 ms. */
function synth(n: number, o: SynthOpts = {}) {
  const video: ProbeVideoSample[] = [];
  const mstp: ProbeMstpSample[] = [];
  for (let i = 0; i < n; i++) {
    const tsMs = 50_000 + (i * 1000) / 30;
    const page = tsMs - 49_900; // capture time on the performance.now() timeline
    const arrival = page + 4 + (i % 3); // worker arrival: 4–6 ms after capture
    const portrait = o.portraitMstpAt !== undefined && i >= o.portraitMstpAt;
    mstp.push({ timestampUs: tsMs * 1000, receivedMs: arrival, codedWidth: portrait ? 1440 : 1920, codedHeight: portrait ? 1920 : 1440, displayWidth: 1920, displayHeight: 1440, format: 'NV12' });
    if (o.dropVideo?.(i)) continue;
    const capture = page + 20 + (o.jitter?.(i) ?? 0) + (o.driftMsPerS ?? 0) * (i / 30); // rVFC captureTime: post-ISP
    const theta = o.theta?.(i) ?? 0;
    const landscape = (theta % 180 !== 0) !== !!o.misoriented?.(i);
    video.push({ captureTime: capture, presentationTime: capture + 10, callbackTime: capture + 12, width: landscape ? 1920 : 1440, height: landscape ? 1440 : 1920, screenAngle: theta });
  }
  return { video, mstp };
}

function grabs(kind: ProbeGrabSample['kind'], syncMs: number, w: number, h: number, n = 10, theta = 0): ProbeGrabSample[] {
  return Array.from({ length: n }, () => ({ kind, syncMs, totalMs: syncMs + 1, width: w, height: h, screenAngle: theta }));
}

describe('analyzeCameraProbe', () => {
  it('a constant rVFC lag shows as an offset, not as jitter or drift; grab costs get verdicts', () => {
    const { video, mstp } = synth(300);
    const g = [...grabs('bitmap', 14, 1440, 1920), ...grabs('bitmapScaled', 6, 960, 1280), ...grabs('videoFrame', 0.3, 1920, 1440)];
    const r = analyzeCameraProbe({ durationMs: 10_000, warmupMs: 2000, video, mstp, grabs: g, mstpError: null, mstpTransport: 'track' });
    expect(r.timing).toMatchObject({ pairs: 300, segments: 1, jitterStdMs: 0, driftMsPerS: 0 });
    expect(r.timing.offsetMedianMs).toBeCloseTo(20 - 49_900, 6);
    expect(r.rotation).toMatchObject({ changes: 0, videoDropped: 0, mstpDropped: 0, videoMisoriented: 0, maxErrorMs: null });
    expect(r.mstp).toMatchObject({ frames: 300, alwaysLandscape: true, codedSizes: ['1920x1440'], transport: 'track' });
    expect(r.mstp.arrivalLatencyJitterMs!.max).toBe(2);
    expect(r.grab.bitmap!.syncMs.p50).toBe(14);
    expect(r.grab.videoFrame!.sizesByTheta).toEqual({ '0': ['1920x1440'] });
    const text = r.summary.join('\n');
    expect(text).toMatch(/small: MSTP timestamps would gain little/);
    expect(text).toMatch(/createImageBitmap\(<video>\): main thread p50 14 ms .*≈14% at 10 Hz\) — material/);
    expect(text).toMatch(/downscaled: main thread p50 6 ms .* — material/);
    expect(text).toMatch(/new VideoFrame\(<video>\): main thread p50 0.3 ms .* — small/);
    expect(text).toMatch(/new VideoFrame\(<video>\) at θ 0° is 1920x1440 while the <video> shows 1440x1920: a raw sensor buffer/);
    expect(text).toMatch(/Rotate the phone/);
  });

  it('a VideoFrame with the same aspect as the <video> is upright for the UI', () => {
    const { video, mstp } = synth(30, { theta: () => 90 });
    const r = analyzeCameraProbe({ durationMs: 1000, video, mstp, grabs: grabs('videoFrame', 0.2, 1920, 1440, 5, 90), mstpError: null, mstpTransport: 'track' });
    expect(r.summary.join('\n')).toMatch(/at θ 90° matches the <video> \(1920x1440\): upright for the UI/);
  });

  it('measures jitter and drift against the capture clock', () => {
    // ±3 ms alternating jitter and 1 ms/s drift
    const { video, mstp } = synth(300, { jitter: (i) => (i % 2 ? 3 : -3), driftMsPerS: 1 });
    const r = analyzeCameraProbe({ durationMs: 10_000, video, mstp, grabs: [], mstpError: null, mstpTransport: 'track' });
    expect(r.timing.pairs).toBe(300);
    expect(r.timing.driftMsPerS!).toBeCloseTo(1, 1);
    expect(r.timing.jitterStdMs!).toBeCloseTo(3, 1); // the injected ±3 ms; drift is not counted again
    expect(r.timing.jitterRangeMs!).toBeCloseTo(6, 1);
    const text = r.summary.join('\n');
    expect(text).toMatch(/material: MSTP timestamps would help/);
    expect(text).toMatch(/Drift 1\.\d+ ms\/s ≈ 30\d ms over a 5-minute scan — material/);
    expect(text).toMatch(/Run the 60 s still probe/);
  });

  it('a timestamp step and transients at a rotation are reported as rotation effects, not drift or jitter', () => {
    // UI rotates at frame 150 (5 s): the rVFC lag steps by +6 ms, frames 150–155 carry +8 ms errors
    // (below half a frame: larger sudden errors cannot be told from dropped frames by timestamps),
    // frames 150–153 have the wrong orientation and 152–153 are dropped (2 misoriented frames remain)
    const { video, mstp } = synth(300, {
      theta: (i) => (i < 150 ? 0 : 90),
      jitter: (i) => (i >= 150 ? 6 : 0) + (i >= 150 && i <= 155 ? 8 : 0),
      misoriented: (i) => i >= 150 && i < 154,
      dropVideo: (i) => i === 152 || i === 153,
    });
    const r = analyzeCameraProbe({ durationMs: 10_000, video, mstp, grabs: [], mstpError: null, mstpTransport: 'track' });
    expect(r.timing.segments).toBe(2);
    expect(r.timing.driftMsPerS).toBe(0); // a one-line fit would report ≈1 ms/s here
    expect(r.timing.jitterStdMs).toBe(0);
    expect(r.timing.pairs).toBeLessThan(298 - 25); // ±500 ms around the rotation are excluded
    expect(r.rotation).toMatchObject({ changes: 1, videoDropped: 2, mstpDropped: 0, videoMisoriented: 2, videoMisorientedSteady: 0 });
    expect(r.rotation.maxErrorMs).toBeCloseTo(8, 6);
    expect(r.video.sizesByTheta).toEqual({ '0': ['1440x1920'], '90': ['1440x1920', '1920x1440'] }); // the first 90° frames are misoriented
    expect(r.summary.join('\n')).toMatch(/Around 1 rotation\(s\): rVFC dropped 2 frame\(s\) \(MSTP 0\), 2 rVFC frame\(s\) had the wrong orientation, worst timestamp error 8 ms/);
  });

  it('does not mispair around a frame MSTP dropped, and flags non-landscape MSTP buffers', () => {
    const { video, mstp } = synth(90, { portraitMstpAt: 80 });
    mstp.splice(40, 1); // dropped by the processor queue
    const r = analyzeCameraProbe({ durationMs: 3000, video, mstp, grabs: [], mstpError: null, mstpTransport: 'stream' });
    expect(r.timing.pairs).toBe(89); // rVFC frame 40 has no partner: not paired with a neighbour
    expect(r.timing.jitterStdMs).toBe(0);
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
    expect(r.grab).toEqual({});
    expect(r.summary[0]).toMatch(/unavailable: MediaStreamTrackProcessor is not available/);
  });

  it('dist', () => {
    expect(dist([])).toBeNull();
    expect(dist([3, 1, 2, NaN])).toEqual({ n: 3, p50: 2, p95: 3, max: 3 });
  });
});
