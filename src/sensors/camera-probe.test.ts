import { describe, expect, it } from 'vitest';
import { rotateRGBA90 } from '../features/image';
import { analyzeCameraProbe, bestQuarterTurn, dist, TURN_THUMB, type ProbeGrabSample, type ProbeMstpSample, type ProbeVideoSample } from './camera-probe';
import { canonicalRotation } from './frame-orientation';

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
    expect(text).toMatch(/rotation unknown \(no confident thumbnail match.*\); it is 1920x1440 in a portrait UI, i.e. sideways there/);
    expect(text).toMatch(/Drift: not enough still time \(longest stretch between rotations 10 s\)/);
    expect(text).toMatch(/Rotate the phone/);
  });

  it('bestQuarterTurn finds the clockwise turns of rotateRGBA90 (the engine convention), also for 4:3 frames squashed square', () => {
    const n = TURN_THUMB;
    // a textured, asymmetric 4:3 image (64×48), and its thumbnails squashed to n×n
    const W = 64;
    const H = 48;
    const data = new Uint8ClampedArray(W * H * 4);
    // smooth random texture (bilinear over a random 7×5 grid): asymmetric, and like a camera image at
    // thumbnail scale (per-pixel noise would not survive the resampling)
    let seed = 12345;
    const grid = Array.from({ length: 35 }, () => {
      seed = (seed * 1103515245 + 12345) >>> 0;
      return seed >>> 24;
    });
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const gx = (x / (W - 1)) * 6;
        const gy = (y / (H - 1)) * 4;
        const x0 = Math.min(5, Math.floor(gx));
        const y0 = Math.min(3, Math.floor(gy));
        const fx = gx - x0;
        const fy = gy - y0;
        const at = (i: number, j: number): number => grid[j * 7 + i];
        const g = (1 - fy) * ((1 - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) + fy * ((1 - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
        data.set([g, g, g, 255], 4 * (y * W + x));
      }
    const squash = (img: { width: number; height: number; data: ArrayLike<number> }): Float32Array => {
      const out = new Float32Array(n * n);
      for (let y = 0; y < n; y++)
        for (let x = 0; x < n; x++) {
          const sx = Math.floor(((x + 0.5) * img.width) / n);
          const sy = Math.floor(((y + 0.5) * img.height) / n);
          out[y * n + x] = img.data[4 * (sy * img.width + sx)];
        }
      return out;
    };
    const upright = squash({ width: W, height: H, data });
    for (let k = 0; k < 4; k++) {
      // the frame is the upright image turned k back (counter-clockwise): it needs k clockwise turns
      const frame = squash(rotateRGBA90({ width: W, height: H, data }, -k));
      const m = bestQuarterTurn(frame, upright, n);
      expect(m.turns, `k=${k}`).toBe(k);
      expect(m.margin).toBeGreaterThan(0.5);
    }
    // a flat image gives no margin
    expect(bestQuarterTurn(new Float32Array(n * n).fill(7), upright, n).margin).toBe(0);
  });

  it('VideoFrame rotation per θ: raw sensor buffer, upright, and votes near rotations ignored', () => {
    const { video, mstp } = synth(300, { theta: (i) => (i < 100 ? 0 : i < 200 ? 90 : 270) });
    const vt = video.map((v) => v.captureTime!);
    const vfGrab = (i: number, turns: number | null): ProbeGrabSample => ({ kind: 'videoFrame', syncMs: 0.2, totalMs: 0.2, width: 1920, height: 1440, screenAngle: video[i].screenAngle, t: vt[i], turns });
    const rawTurns = (i: number): number => canonicalRotation(video[i].screenAngle, 90);
    // raw buffers: θ0 → 1, θ90 → 0, θ270 → 2; one wrong vote right after the rotation at frame 100 is ignored
    const raw = [5, 20, 50, 130, 150, 170, 230, 250, 280].map((i) => vfGrab(i, rawTurns(i)));
    raw.push(vfGrab(101, 3));
    const r = analyzeCameraProbe({ durationMs: 10_000, video, mstp, grabs: raw, mstpError: null, mstpTransport: 'track' });
    expect(r.grab.videoFrame!.turnsByTheta).toEqual({ '0': { turns: 1, n: 3, agree: 3 }, '90': { turns: 0, n: 3, agree: 3 }, '270': { turns: 2, n: 3, agree: 3 } });
    expect(r.summary.join('\n')).toMatch(/θ 0° → 1 \(3\/3\), θ 90° → 0 \(3\/3\), θ 270° → 2 \(3\/3\) — a raw sensor buffer \(upright for 90°\) at every angle measured\./);
    // pixels rotated upright for the UI
    const up = analyzeCameraProbe({ durationMs: 10_000, video, mstp, grabs: [5, 150, 250].map((i) => vfGrab(i, 0)), mstpError: null, mstpTransport: 'track' });
    expect(up.summary.join('\n')).toMatch(/upright for the UI at every angle measured/);
    // raw except at 270 (the dangerous case: a raw flag would turn it 180°)
    const mixed = analyzeCameraProbe({ durationMs: 10_000, video, mstp, grabs: [vfGrab(5, 1), vfGrab(150, 0), vfGrab(250, 0)], mstpError: null, mstpTransport: 'track' });
    expect(mixed.summary.join('\n')).toMatch(/NEITHER consistently raw nor upright: raw at 0°, 90°/);
  });

  it('measures jitter and drift against the capture clock', () => {
    // ±3 ms alternating jitter and 1 ms/s drift
    const { video, mstp } = synth(900, { jitter: (i) => (i % 2 ? 3 : -3), driftMsPerS: 0.5 });
    const r = analyzeCameraProbe({ durationMs: 30_000, video, mstp, grabs: [], mstpError: null, mstpTransport: 'track' });
    expect(r.timing.pairs).toBe(900);
    expect(r.timing.longestSteadyS).toBeCloseTo(30, 0);
    expect(r.timing.driftMsPerS!).toBeCloseTo(0.5, 1);
    expect(r.timing.jitterStdMs!).toBeCloseTo(3, 1); // the injected ±3 ms; drift is not counted again
    expect(r.timing.jitterRangeMs!).toBeCloseTo(6, 1);
    const text = r.summary.join('\n');
    expect(text).toMatch(/material: MSTP timestamps would help/);
    expect(text).toMatch(/Drift 0\.\d+ ms\/s ≈ 15\d ms over a 5-minute scan — material/);
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
