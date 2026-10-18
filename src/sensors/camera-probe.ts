/**
 * Camera timing probe (Diagnostics): measures, on the device, whether reading camera frames with
 * MediaStreamTrackProcessor (docs/research/ios-camera.md §7 ladder step A) would improve on the
 * capture path in use (rVFC + createImageBitmap(<video>), ./camera.ts) — before building it.
 *
 * For `durationMs` it records, on the same camera track:
 *  - every rVFC callback (captureTime / presentationTime, frame size, UI rotation θ);
 *  - every frame a clone of the track delivers to a MediaStreamTrackProcessor in a worker
 *    (src/workers/camera-probe.worker.ts): VideoFrame.timestamp (capture clock), arrival time,
 *    coded / display size, format. Frames are closed at once (nothing is converted);
 *  - createImageBitmap(<video>) at 10 Hz, as FramePump does: time spent synchronously on the
 *    main thread and until the bitmap resolves.
 * `analyzeCameraProbe` (pure) pairs rVFC frames with MSTP frames and reports how much rVFC
 * captureTime jitters and drifts against the capture clock (what MSTP timestamps would remove;
 * a CONSTANT lag is already absorbed by the engine's camera–IMU time-offset estimate), whether
 * MSTP buffers stay sensor-landscape in every UI rotation, and what the main-thread grab costs.
 */
import type { CameraCapture } from './camera';
import { screenAngle } from './frame-orientation';

export interface ProbeVideoSample {
  /** rVFC captureTime (ms, performance.now()), when provided. */
  captureTime: number | null;
  presentationTime: number | null;
  callbackTime: number;
  width: number;
  height: number;
  screenAngle: number;
}

export interface ProbeMstpSample {
  /** VideoFrame.timestamp (µs, capture clock). */
  timestampUs: number;
  /** Arrival in the worker, on the MAIN thread's performance.now() timeline. */
  receivedMs: number;
  codedWidth: number;
  codedHeight: number;
  displayWidth: number;
  displayHeight: number;
  format: string | null;
}

export interface ProbeGrabSample {
  /** Synchronous part of the createImageBitmap(<video>) call (main thread blocked). */
  syncMs: number;
  /** Until the bitmap resolved. */
  totalMs: number;
}

export interface ProbeInput {
  durationMs: number;
  video: ProbeVideoSample[];
  mstp: ProbeMstpSample[];
  grabs: ProbeGrabSample[];
  /** Why the MSTP side recorded nothing (null = it ran). */
  mstpError: string | null;
  /** 'track' = transferred to the worker (iOS), 'stream' = main-thread processor (Chromium). */
  mstpTransport: 'track' | 'stream' | null;
}

export interface Dist {
  n: number;
  p50: number;
  p95: number;
  max: number;
}

export interface CameraProbeReport {
  durationMs: number;
  video: { frames: number; fps: number; timeSource: 'captureTime' | 'presentationTime' | 'callback'; intervalMs: Dist | null; sizesByTheta: Record<string, string[]> };
  mstp: {
    available: boolean;
    error: string | null;
    transport: 'track' | 'stream' | null;
    frames: number;
    fps: number;
    intervalMs: Dist | null;
    codedSizes: string[];
    formats: string[];
    /** Every buffer had codedWidth > codedHeight (raw sensor-landscape) — null without frames. */
    alwaysLandscape: boolean | null;
    /** Arrival − capture time above its minimum: the jitter of "time the frame reached the page". */
    arrivalLatencyJitterMs: Dist | null;
  };
  /** rVFC captureTime against the MSTP capture clock, over paired frames. */
  timing: {
    pairs: number;
    /** Median (captureTime − timestamp) — an arbitrary constant; the spread around it matters. */
    offsetMedianMs: number | null;
    /** Standard deviation of (captureTime − timestamp) around its drift line. */
    jitterStdMs: number | null;
    /** p95 − p5 of the same residuals. */
    jitterRangeMs: number | null;
    /** Least-squares slope of (captureTime − timestamp) against time (ms per s). */
    driftMsPerS: number | null;
  };
  grab: { syncMs: Dist | null; totalMs: Dist | null };
  /** One-line conclusions for the UI. */
  summary: string[];
}

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return NaN;
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))));
  return sorted[i];
}

export function dist(values: number[]): Dist | null {
  const v = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (v.length === 0) return null;
  return { n: v.length, p50: quantile(v, 0.5), p95: quantile(v, 0.95), max: v[v.length - 1] };
}

function diffs(a: number[]): number[] {
  const out: number[] = [];
  for (let i = 1; i < a.length; i++) out.push(a[i] - a[i - 1]);
  return out;
}

const round = (x: number, d = 2): number => Math.round(x * 10 ** d) / 10 ** d;

/** Pure analysis of the recorded samples (see the file header). */
export function analyzeCameraProbe(inp: ProbeInput): CameraProbeReport {
  const secs = Math.max(1e-3, inp.durationMs / 1000);
  const v = inp.video;
  const timeSource: CameraProbeReport['video']['timeSource'] = v.length && v.every((s) => s.captureTime !== null)
    ? 'captureTime'
    : v.length && v.every((s) => s.presentationTime !== null)
      ? 'presentationTime'
      : 'callback';
  const vt = v.map((s) => (timeSource === 'captureTime' ? s.captureTime! : timeSource === 'presentationTime' ? s.presentationTime! : s.callbackTime));
  const sizesByTheta: Record<string, string[]> = {};
  for (const s of v) {
    const sizes = (sizesByTheta[String(s.screenAngle)] ??= []);
    const size = `${s.width}x${s.height}`;
    if (!sizes.includes(size)) sizes.push(size);
  }

  const m = inp.mstp;
  const mts = m.map((s) => s.timestampUs / 1000);
  const lat = m.map((s, i) => s.receivedMs - mts[i]);
  const minLat = lat.length ? Math.min(...lat) : NaN;

  // Pairing: a coarse offset o = median(captureTime − timestamp) over "arrived closest in time"
  // matches, then each rVFC frame is paired with the MSTP frame whose timestamp + o is nearest and
  // kept within half a frame interval (frames are ~33 ms apart, so the pairing is unambiguous for
  // jitter below that; a pairing shifted by one frame for ALL frames only shifts the offset).
  // Drift = least-squares slope of d = captureTime − timestamp over time; jitter = the residuals
  // around that line (so drift is not counted twice).
  const pairs: { t: number; d: number }[] = [];
  if (timeSource === 'captureTime' && m.length) {
    const coarse: number[] = [];
    let j = 0;
    for (const s of v) {
      const ct = s.captureTime!;
      while (j + 1 < m.length && Math.abs(m[j + 1].receivedMs - ct) <= Math.abs(m[j].receivedMs - ct)) j++;
      coarse.push(ct - mts[j]);
    }
    const halfFrame = 0.5 * (dist(diffs(mts))?.p50 ?? 1000 / 30);
    const pairWith = (o: number): { t: number; d: number }[] => {
      const out: { t: number; d: number }[] = [];
      let k = 0;
      for (const s of v) {
        const target = s.captureTime! - o;
        while (k + 1 < mts.length && Math.abs(mts[k + 1] - target) <= Math.abs(mts[k] - target)) k++;
        if (Math.abs(mts[k] - target) < halfFrame) out.push({ t: mts[k], d: s.captureTime! - mts[k] });
      }
      return out;
    };
    const medianD = (ds: number[]): number =>
      quantile(
        [...ds].sort((a, b) => a - b),
        0.5,
      );
    // the coarse matches can straddle a frame boundary when the rVFC lag is near half a frame:
    // re-centre on the first pass's median, then keep whichever one-frame shift pairs the most
    // frames (a shift of the whole pairing changes only the arbitrary offset)
    const first = pairWith(medianD(coarse));
    if (first.length) {
      const o = medianD(first.map((p) => p.d));
      let best: { t: number; d: number }[] = [];
      for (const c of [o, o - 2 * halfFrame, o + 2 * halfFrame]) {
        const ps = pairWith(c);
        if (ps.length > best.length) best = ps;
      }
      pairs.push(...best);
    }
  }
  let offsetMedianMs: number | null = null;
  let jitterStdMs: number | null = null;
  let jitterRangeMs: number | null = null;
  let driftMsPerS: number | null = null;
  if (pairs.length >= 2) {
    offsetMedianMs = round(
      quantile(
        pairs.map((p) => p.d).sort((a, b) => a - b),
        0.5,
      ),
    );
    const mt = pairs.reduce((a, p) => a + p.t, 0) / pairs.length;
    const md = pairs.reduce((a, p) => a + p.d, 0) / pairs.length;
    let num = 0;
    let den = 0;
    for (const p of pairs) {
      num += (p.t - mt) * (p.d - md);
      den += (p.t - mt) ** 2;
    }
    const slope = den > 0 ? num / den : 0;
    driftMsPerS = round(slope * 1000, 3);
    const res = pairs.map((p) => p.d - (md + slope * (p.t - mt))).sort((a, b) => a - b);
    jitterStdMs = round(Math.sqrt(res.reduce((a, r) => a + r * r, 0) / res.length));
    jitterRangeMs = round(quantile(res, 0.95) - quantile(res, 0.05));
  }

  const codedSizes = [...new Set(m.map((s) => `${s.codedWidth}x${s.codedHeight}`))];
  const report: CameraProbeReport = {
    durationMs: inp.durationMs,
    video: { frames: v.length, fps: round(v.length / secs, 1), timeSource, intervalMs: dist(diffs(vt)), sizesByTheta },
    mstp: {
      available: inp.mstpError === null,
      error: inp.mstpError,
      transport: inp.mstpTransport,
      frames: m.length,
      fps: round(m.length / secs, 1),
      intervalMs: dist(diffs(mts)),
      codedSizes,
      formats: [...new Set(m.map((s) => s.format ?? 'unknown'))],
      alwaysLandscape: m.length ? m.every((s) => s.codedWidth > s.codedHeight) : null,
      arrivalLatencyJitterMs: dist(lat.map((l) => l - minLat)),
    },
    timing: { pairs: pairs.length, offsetMedianMs, jitterStdMs, jitterRangeMs, driftMsPerS },
    grab: { syncMs: dist(inp.grabs.map((g) => g.syncMs)), totalMs: dist(inp.grabs.map((g) => g.totalMs)) },
    summary: [],
  };
  report.summary = summarize(report);
  return report;
}

function summarize(r: CameraProbeReport): string[] {
  const out: string[] = [];
  if (!r.mstp.available) out.push(`MediaStreamTrackProcessor unavailable: ${r.mstp.error}`);
  else if (!r.mstp.frames) out.push('MediaStreamTrackProcessor delivered no frames.');
  const t = r.timing;
  if (t.jitterStdMs !== null) {
    const material = (t.jitterRangeMs ?? 0) >= 5 || Math.abs(t.driftMsPerS ?? 0) >= 0.5;
    out.push(
      `rVFC captureTime vs capture clock: σ ${t.jitterStdMs} ms, p5–p95 ${t.jitterRangeMs} ms, drift ${t.driftMsPerS} ms/s over ${t.pairs} frames — ` +
        (material ? 'material: MSTP timestamps would help.' : 'small: MSTP timestamps would gain little.'),
    );
  } else if (r.mstp.frames && r.video.timeSource !== 'captureTime') out.push(`rVFC gives no captureTime here (${r.video.timeSource}): MSTP timestamps would be the only capture clock.`);
  if (r.mstp.alwaysLandscape !== null) out.push(r.mstp.alwaysLandscape ? `MSTP buffers stayed landscape (${r.mstp.codedSizes.join(', ')}).` : `MSTP buffers were NOT always landscape (${r.mstp.codedSizes.join(', ')}).`);
  const angles = Object.keys(r.video.sizesByTheta);
  if (angles.length < 2) out.push('Rotate the phone during the probe to test orientation.');
  const g = r.grab.syncMs;
  if (g) out.push(`createImageBitmap(<video>) blocks the main thread p50 ${round(g.p50)} ms / p95 ${round(g.p95)} ms (resolves p50 ${round(r.grab.totalMs!.p50)} ms).`);
  return out;
}

// ---------------------------------------------------------------------------------------------
// Runner (browser)
// ---------------------------------------------------------------------------------------------

/** Worker side of the probe (src/workers/camera-probe.worker.ts). */
export type ProbeWorkerRequest = { type: 'start'; track?: MediaStreamTrack; readable?: ReadableStream<unknown> } | { type: 'stop' };
export type ProbeWorkerEvent =
  | { type: 'started'; timeOrigin: number }
  | { type: 'error'; message: string }
  | { type: 'samples'; samples: Omit<ProbeMstpSample, 'receivedMs'>[]; receivedMs: number[] };

export interface ProbeWorker {
  postMessage(msg: ProbeWorkerRequest, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent<ProbeWorkerEvent>) => void) | null;
  onerror: ((e: ErrorEvent) => void) | null;
  terminate(): void;
}

export interface ProbeOptions {
  durationMs?: number;
  /** createImageBitmap(<video>) rate (default 10 Hz, as FramePump). */
  grabHz?: number;
  createWorker?: () => ProbeWorker;
  /** Remaining ms, at each grab. */
  onProgress?: (remainingMs: number) => void;
  /** Stop early (e.g. the screen was left); the report covers the time measured. */
  isCancelled?: () => boolean;
}

/**
 * Run the probe on a RUNNING CameraCapture (its <video> must be playing). The camera is left
 * running; the worker's track clone is stopped.
 */
export async function runCameraTimingProbe(camera: CameraCapture, opts: ProbeOptions = {}): Promise<CameraProbeReport> {
  const durationMs = opts.durationMs ?? 10_000;
  const track = camera.track;
  if (!track || track.readyState !== 'live') throw new Error('Camera timing probe: the camera is not running');
  const video: ProbeVideoSample[] = [];
  const grabs: ProbeGrabSample[] = [];
  let mstpError: string | null = null;
  let transport: ProbeInput['mstpTransport'] = null;
  let worker: ProbeWorker | null = null;
  let clone: MediaStreamTrack | null = null;
  let localClone: MediaStreamTrack | null = null;
  let originDelta = 0;
  let started: Promise<void> = Promise.resolve();
  let samples: Promise<ProbeMstpSample[]> = Promise.resolve([]);

  try {
    worker = (opts.createWorker ?? defaultProbeWorker)();
    const w = worker;
    let resolveStarted!: () => void;
    let resolveSamples!: (s: ProbeMstpSample[]) => void;
    started = new Promise((r) => (resolveStarted = r));
    samples = new Promise((r) => (resolveSamples = r));
    w.onmessage = (e) => {
      const ev = e.data;
      if (ev.type === 'started') {
        originDelta = ev.timeOrigin - performance.timeOrigin; // both read now
        resolveStarted();
      } else if (ev.type === 'error') {
        mstpError = ev.message;
        resolveStarted();
        resolveSamples([]);
      } else resolveSamples(ev.samples.map((s, i) => ({ ...s, receivedMs: ev.receivedMs[i] + originDelta })));
    };
    w.onerror = (e) => {
      e.preventDefault?.();
      mstpError = `probe worker error: ${e.message || 'script error'}`;
      resolveStarted();
      resolveSamples([]);
    };
    clone = track.clone();
    try {
      w.postMessage({ type: 'start', track: clone }, [clone as unknown as Transferable]);
      transport = 'track';
    } catch (e) {
      // Chromium: tracks are not transferable; transfer a main-thread processor's stream instead
      const P = (globalThis as { MediaStreamTrackProcessor?: new (i: { track: MediaStreamTrack }) => { readable: ReadableStream<unknown> } }).MediaStreamTrackProcessor;
      if (typeof P !== 'function') throw e;
      const readable = new P({ track: clone }).readable;
      w.postMessage({ type: 'start', readable }, [readable as unknown as Transferable]);
      localClone = clone;
      transport = 'stream';
    }
    await Promise.race([started, sleep(3000)]);
  } catch (e) {
    mstpError ??= `MSTP path unavailable: ${(e as Error)?.message ?? e}`;
    clone?.stop();
  }

  // main-thread side: rVFC samples and 10 Hz grabs, for durationMs
  const v = camera.video as (HTMLVideoElement & { requestVideoFrameCallback?: (cb: (n: number, md: Record<string, number>) => void) => number; cancelVideoFrameCallback?: (h: number) => void }) | null;
  let rvfc = 0;
  let running = true;
  const onVideoFrame = (_now: number, md: Record<string, number>): void => {
    if (!running || !v?.requestVideoFrameCallback) return;
    rvfc = v.requestVideoFrameCallback(onVideoFrame);
    video.push({
      captureTime: typeof md.captureTime === 'number' ? md.captureTime : null,
      presentationTime: typeof md.presentationTime === 'number' ? md.presentationTime : null,
      callbackTime: performance.now(),
      width: md.width || v.videoWidth,
      height: md.height || v.videoHeight,
      screenAngle: screenAngle(),
    });
  };
  if (v?.requestVideoFrameCallback) rvfc = v.requestVideoFrameCallback(onVideoFrame);
  const t0 = performance.now();
  const grabEvery = 1000 / (opts.grabHz ?? 10);
  while (performance.now() - t0 < durationMs && !opts.isCancelled?.()) {
    if (v && v.readyState >= 2) {
      const g0 = performance.now();
      const p = createImageBitmap(v);
      const g1 = performance.now();
      try {
        const bmp = await p;
        grabs.push({ syncMs: g1 - g0, totalMs: performance.now() - g0 });
        bmp.close();
      } catch {
        /* not counted */
      }
    }
    opts.onProgress?.(Math.max(0, durationMs - (performance.now() - t0)));
    await sleep(Math.max(0, grabEvery - (performance.now() - t0) % grabEvery));
  }
  running = false;
  if (v?.cancelVideoFrameCallback) v.cancelVideoFrameCallback(rvfc);

  let mstp: ProbeMstpSample[] = [];
  if (worker) {
    if (mstpError === null) {
      worker.postMessage({ type: 'stop' });
      mstp = await Promise.race([samples, sleep(3000).then(() => [] as ProbeMstpSample[])]);
    }
    worker.terminate();
  }
  localClone?.stop();
  const measuredMs = Math.min(durationMs, performance.now() - t0);
  return analyzeCameraProbe({ durationMs: measuredMs, video, mstp, grabs, mstpError, mstpTransport: mstpError === null ? transport : null });
}

function defaultProbeWorker(): ProbeWorker {
  return new Worker(new URL('../workers/camera-probe.worker.ts', import.meta.url), { type: 'module', name: 'camera-probe' }) as unknown as ProbeWorker;
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
