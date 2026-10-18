/**
 * Camera timing probe (Diagnostics): measures, on the device, whether reading camera frames another
 * way would improve on the capture path in use (rVFC + createImageBitmap(<video>), ./camera.ts)
 * before building it (docs/research/ios-camera.md §7 ladder steps A/B).
 *
 * After `warmupMs` (camera latency settles) it records for `durationMs`, on the same camera track:
 *  - every rVFC callback (captureTime / presentationTime, frame size, UI rotation θ);
 *  - every frame a clone of the track delivers to a MediaStreamTrackProcessor in a worker
 *    (src/workers/camera-probe.worker.ts): VideoFrame.timestamp (capture clock), arrival time,
 *    coded / display size, format. Frames are closed at once (nothing is converted);
 *  - 10 grabs per second (FramePump's rate), cycling through createImageBitmap(<video>) at full
 *    size, the same downscaled to `scaledMaxSide`, and new VideoFrame(<video>) (step B: an
 *    IOSurface reference handed to a worker): time on the main thread, until resolved, and the
 *    size per θ (is a VideoFrame of the <video> a raw sensor-landscape buffer?).
 *
 * `analyzeCameraProbe` (pure) pairs rVFC frames with MSTP frames. Frames within ±500 ms of a UI
 * rotation (which the engine drops anyway) are reported separately (dropped frames, wrongly
 * oriented frames, worst timestamp error); on the rest it reports how much rVFC captureTime
 * jitters and drifts against the capture clock — drift as one slope shared by the segments
 * between rotations, so a step at a rotation is not mistaken for drift. A CONSTANT lag does not
 * matter: the engine's camera–IMU time-offset estimate absorbs it; drift does, over a long scan.
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

export type GrabKind = 'bitmap' | 'bitmapScaled' | 'videoFrame';

export interface ProbeGrabSample {
  kind: GrabKind;
  /** Synchronous part of the call (main thread blocked). */
  syncMs: number;
  /** Until the result was available. */
  totalMs: number;
  /** Result size (VideoFrame: displayWidth × displayHeight). */
  width: number;
  height: number;
  screenAngle: number;
}

export interface ProbeInput {
  durationMs: number;
  warmupMs?: number;
  video: ProbeVideoSample[];
  mstp: ProbeMstpSample[];
  grabs: ProbeGrabSample[];
  /** Why the MSTP side recorded nothing (null = it ran). */
  mstpError: string | null;
  /** 'track' = transferred to the worker (iOS), 'stream' = main-thread processor (Chromium). */
  mstpTransport: 'track' | 'stream' | null;
  /** Grabs per second (default 10): for the main-thread share. */
  grabHz?: number;
}

export interface Dist {
  n: number;
  p50: number;
  p95: number;
  max: number;
}

export interface GrabStats {
  syncMs: Dist;
  totalMs: Dist;
  sizesByTheta: Record<string, string[]>;
}

export interface CameraProbeReport {
  durationMs: number;
  warmupMs: number;
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
  /** rVFC captureTime against the MSTP capture clock, over paired frames away from rotations. */
  timing: {
    pairs: number;
    /** Segments between UI rotations that contributed pairs. */
    segments: number;
    /** Median (captureTime − timestamp) — an arbitrary constant; the spread around it matters. */
    offsetMedianMs: number | null;
    /** Standard deviation of (captureTime − timestamp) around the drift line of its segment. */
    jitterStdMs: number | null;
    /** p95 − p5 of the same residuals. */
    jitterRangeMs: number | null;
    /** Slope of (captureTime − timestamp) against time shared by all segments (ms per s). */
    driftMsPerS: number | null;
  };
  /** Within ±ROTATION_WINDOW_MS of each UI rotation. */
  rotation: {
    changes: number;
    /** Frames missing from the rVFC / MSTP sequences (gaps over 1.5 frame intervals). */
    videoDropped: number;
    mstpDropped: number;
    /** rVFC frames whose aspect contradicts the UI rotation (all / away from rotations). */
    videoMisoriented: number;
    videoMisorientedSteady: number;
    /** Largest |timestamp error| of paired frames near rotations, against the steady fit. */
    maxErrorMs: number | null;
  };
  grab: Partial<Record<GrabKind, GrabStats>>;
  /** One-line conclusions for the UI. */
  summary: string[];
}

/** Frames this close to a UI rotation are treated as rotation transients (the engine drops them). */
export const ROTATION_WINDOW_MS = 500;
/** Grab cost on the main thread above which it is worth moving off it (ms per grab). */
export const MATERIAL_GRAB_MS = 4;
/** |drift| above which it matters over a scan (ms/s: 0.1 ⇒ 30 ms over 5 min). */
export const MATERIAL_DRIFT_MS_PER_S = 0.1;
/** p5–p95 timestamp spread above which MSTP timestamps would matter (ms). */
export const MATERIAL_JITTER_MS = 5;

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
const median = (xs: number[]): number =>
  quantile(
    [...xs].sort((a, b) => a - b),
    0.5,
  );

/**
 * Frames missing from a time sequence: gaps over 1.5 nominal intervals, counted in intervals. A
 * sudden timestamp error above half an interval reads as a missing frame (indistinguishable).
 */
function missing(times: number[], period: number, keep: (t: number) => boolean = () => true): number {
  let n = 0;
  for (let i = 1; i < times.length; i++) {
    const g = times[i] - times[i - 1];
    if (g > 1.5 * period && (keep(times[i]) || keep(times[i - 1]))) n += Math.round(g / period) - 1;
  }
  return n;
}

function addSize(map: Record<string, string[]>, theta: number, w: number, h: number): void {
  const sizes = (map[String(theta)] ??= []);
  const size = `${w}x${h}`;
  if (!sizes.includes(size)) sizes.push(size);
}

/** Aspect contradicts the UI rotation (portrait UI with a landscape frame or vice versa). */
function misoriented(w: number, h: number, theta: number): boolean {
  const portraitUI = ((((theta / 90) % 2) + 2) % 2) === 0;
  return portraitUI ? w > h : h > w;
}

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
  for (const s of v) addSize(sizesByTheta, s.screenAngle, s.width, s.height);

  // UI rotations: the frame time (same clock as the pairs' captureTime) of the first frame seen at a new θ
  const changes: number[] = [];
  for (let i = 1; i < v.length; i++) if (v[i].screenAngle !== v[i - 1].screenAngle) changes.push(vt[i]);
  const nearRotation = (t: number): boolean => changes.some((c) => Math.abs(t - c) <= ROTATION_WINDOW_MS);
  const segmentOf = (t: number): number => changes.filter((c) => c <= t).length;

  const m = inp.mstp;
  const mts = m.map((s) => s.timestampUs / 1000);
  const lat = m.map((s, i) => s.receivedMs - mts[i]);
  const minLat = lat.length ? Math.min(...lat) : NaN;
  const mPeriod = dist(diffs(mts))?.p50 ?? 1000 / 30;

  // Pairing: a coarse offset o = median(captureTime − timestamp) over "arrived closest in time"
  // matches, then each rVFC frame is paired with the MSTP frame whose timestamp + o is nearest and
  // kept within half a frame interval (frames are ~33 ms apart, so the pairing is unambiguous for
  // errors below that; a pairing shifted by one frame for ALL frames only shifts the offset).
  type Pair = { t: number; d: number; ct: number };
  let pairs: Pair[] = [];
  if (timeSource === 'captureTime' && m.length) {
    const coarse: number[] = [];
    let j = 0;
    for (const s of v) {
      const ct = s.captureTime!;
      while (j + 1 < m.length && Math.abs(m[j + 1].receivedMs - ct) <= Math.abs(m[j].receivedMs - ct)) j++;
      coarse.push(ct - mts[j]);
    }
    const halfFrame = 0.5 * mPeriod;
    const pairWith = (o: number): Pair[] => {
      const out: Pair[] = [];
      let k = 0;
      for (const s of v) {
        const ct = s.captureTime!;
        const target = ct - o;
        while (k + 1 < mts.length && Math.abs(mts[k + 1] - target) <= Math.abs(mts[k] - target)) k++;
        if (Math.abs(mts[k] - target) < halfFrame) out.push({ t: mts[k], d: ct - mts[k], ct });
      }
      return out;
    };
    // the coarse matches can straddle a frame boundary when the rVFC lag is near half a frame:
    // re-centre on the first pass's median, then keep whichever one-frame shift pairs the most
    // frames (a shift of the whole pairing changes only the arbitrary offset)
    const first = pairWith(median(coarse));
    if (first.length) {
      const o = median(first.map((p) => p.d));
      for (const c of [o, o - 2 * halfFrame, o + 2 * halfFrame]) {
        const ps = pairWith(c);
        if (ps.length > pairs.length) pairs = ps;
      }
    }
  }

  // Drift: one slope shared by the segments between rotations (each segment centred on its own
  // mean), fitted on the frames away from rotations; jitter = residuals around it.
  const steady = pairs.filter((p) => !nearRotation(p.ct));
  const bySeg = new Map<number, Pair[]>();
  for (const p of steady) {
    const k = segmentOf(p.ct);
    if (!bySeg.has(k)) bySeg.set(k, []);
    bySeg.get(k)!.push(p);
  }
  const segMean = new Map<number, { t: number; d: number }>();
  let num = 0;
  let den = 0;
  for (const [k, ps] of bySeg) {
    const mt = ps.reduce((a, p) => a + p.t, 0) / ps.length;
    const md = ps.reduce((a, p) => a + p.d, 0) / ps.length;
    segMean.set(k, { t: mt, d: md });
    for (const p of ps) {
      num += (p.t - mt) * (p.d - md);
      den += (p.t - mt) ** 2;
    }
  }
  const slope = den > 0 ? num / den : 0;
  const residual = (p: Pair): number | null => {
    const sm = segMean.get(segmentOf(p.ct));
    return sm ? p.d - (sm.d + slope * (p.t - sm.t)) : null;
  };
  let offsetMedianMs: number | null = null;
  let jitterStdMs: number | null = null;
  let jitterRangeMs: number | null = null;
  let driftMsPerS: number | null = null;
  if (steady.length >= 2) {
    offsetMedianMs = round(median(steady.map((p) => p.d)));
    driftMsPerS = den > 0 ? round(slope * 1000, 3) : null;
    const res = steady.map((p) => residual(p)!).sort((a, b) => a - b);
    jitterStdMs = round(Math.sqrt(res.reduce((a, r) => a + r * r, 0) / res.length));
    jitterRangeMs = round(quantile(res, 0.95) - quantile(res, 0.05));
  }
  const transientErrors = pairs
    .filter((p) => nearRotation(p.ct))
    .map((p) => residual(p))
    .filter((r): r is number => r !== null)
    .map(Math.abs);

  const vPeriod = dist(diffs(vt))?.p50 ?? 1000 / 30;
  // MSTP times are on the capture clock: map them onto the page timeline with the arrival offset
  const toPage = Number.isFinite(minLat) ? minLat : 0;
  const rotation: CameraProbeReport['rotation'] = {
    changes: changes.length,
    videoDropped: missing(vt, vPeriod, nearRotation),
    mstpDropped: missing(mts, mPeriod, (t) => nearRotation(t + toPage)),
    videoMisoriented: v.filter((s) => misoriented(s.width, s.height, s.screenAngle)).length,
    videoMisorientedSteady: v.filter((s, i) => misoriented(s.width, s.height, s.screenAngle) && !nearRotation(vt[i])).length,
    maxErrorMs: transientErrors.length ? round(Math.max(...transientErrors)) : null,
  };

  const grab: CameraProbeReport['grab'] = {};
  for (const kind of ['bitmap', 'bitmapScaled', 'videoFrame'] as const) {
    const gs = inp.grabs.filter((g) => g.kind === kind);
    if (!gs.length) continue;
    const sizes: Record<string, string[]> = {};
    for (const g of gs) addSize(sizes, g.screenAngle, g.width, g.height);
    grab[kind] = { syncMs: dist(gs.map((g) => g.syncMs))!, totalMs: dist(gs.map((g) => g.totalMs))!, sizesByTheta: sizes };
  }

  const codedSizes = [...new Set(m.map((s) => `${s.codedWidth}x${s.codedHeight}`))];
  const report: CameraProbeReport = {
    durationMs: inp.durationMs,
    warmupMs: inp.warmupMs ?? 0,
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
    timing: { pairs: steady.length, segments: bySeg.size, offsetMedianMs, jitterStdMs, jitterRangeMs, driftMsPerS },
    rotation,
    grab,
    summary: [],
  };
  report.summary = summarize(report, inp.grabHz ?? 10);
  return report;
}

const GRAB_LABEL: Record<GrabKind, string> = {
  bitmap: 'createImageBitmap(<video>)',
  bitmapScaled: 'createImageBitmap(<video>) downscaled',
  videoFrame: 'new VideoFrame(<video>)',
};

function summarize(r: CameraProbeReport, grabHz: number): string[] {
  const out: string[] = [];
  if (!r.mstp.available) out.push(`MediaStreamTrackProcessor unavailable: ${r.mstp.error}`);
  else if (!r.mstp.frames) out.push('MediaStreamTrackProcessor delivered no frames.');
  const t = r.timing;
  if (t.jitterStdMs !== null) {
    const jitterMaterial = (t.jitterRangeMs ?? 0) >= MATERIAL_JITTER_MS;
    out.push(
      `Timestamp jitter away from rotations: σ ${t.jitterStdMs} ms, p5–p95 ${t.jitterRangeMs} ms over ${t.pairs} frames — ` +
        (jitterMaterial ? 'material: MSTP timestamps would help.' : 'small: MSTP timestamps would gain little.'),
    );
    if (t.driftMsPerS !== null) {
      const per5min = Math.abs(t.driftMsPerS) * 300;
      out.push(
        `Drift ${t.driftMsPerS} ms/s ≈ ${round(per5min, 0)} ms over a 5-minute scan — ` +
          (Math.abs(t.driftMsPerS) >= MATERIAL_DRIFT_MS_PER_S
            ? 'material: the engine models one constant camera–IMU offset per session.'
            : 'small.') +
          (r.durationMs < 30_000 ? ' (Run the 60 s still probe for a reliable drift.)' : ''),
      );
    }
  } else if (r.mstp.frames && r.video.timeSource !== 'captureTime') out.push(`rVFC gives no captureTime here (${r.video.timeSource}): MSTP timestamps would be the only capture clock.`);
  const rot = r.rotation;
  if (rot.changes) {
    out.push(
      `Around ${rot.changes} rotation(s): rVFC dropped ${rot.videoDropped} frame(s) (MSTP ${rot.mstpDropped}), ${rot.videoMisoriented} rVFC frame(s) had the wrong orientation` +
        (rot.videoMisorientedSteady ? ` (${rot.videoMisorientedSteady} outside the ±${ROTATION_WINDOW_MS} ms window!)` : '') +
        (rot.maxErrorMs !== null ? `, worst timestamp error ${rot.maxErrorMs} ms.` : '.'),
    );
  }
  if (r.mstp.alwaysLandscape !== null) out.push(r.mstp.alwaysLandscape ? `MSTP buffers stayed landscape (${r.mstp.codedSizes.join(', ')}).` : `MSTP buffers were NOT always landscape (${r.mstp.codedSizes.join(', ')}).`);
  if (Object.keys(r.video.sizesByTheta).length < 2 && r.durationMs < 30_000) out.push('Rotate the phone during the probe to test orientation.');
  for (const kind of ['bitmap', 'bitmapScaled', 'videoFrame'] as const) {
    const g = r.grab[kind];
    if (!g) continue;
    const share = round((g.syncMs.p50 * grabHz) / 10, 1); // % of the main thread at grabHz
    out.push(
      `${GRAB_LABEL[kind]}: main thread p50 ${round(g.syncMs.p50)} ms / p95 ${round(g.syncMs.p95)} ms (≈${share}% at ${grabHz} Hz) — ` +
        (g.syncMs.p50 >= MATERIAL_GRAB_MS ? 'material.' : 'small.'),
    );
  }
  // a VideoFrame of the <video> vs the <video> frames themselves (upright for the UI) at the same θ
  const vf = r.grab.videoFrame;
  if (vf) {
    const landscape = (size: string): boolean => {
      const [w, h] = size.split('x').map(Number);
      return w > h;
    };
    for (const [theta, sizes] of Object.entries(vf.sizesByTheta)) {
      const shown = r.video.sizesByTheta[theta] ?? [];
      if (!shown.length || shown.every(landscape) !== shown.some(landscape)) continue; // mixed: a rotation
      const differs = sizes.every((sz) => landscape(sz) !== landscape(shown[0]));
      const same = sizes.every((sz) => landscape(sz) === landscape(shown[0]));
      if (differs) out.push(`new VideoFrame(<video>) at θ ${theta}° is ${sizes.join(', ')} while the <video> shows ${shown[0]}: a raw sensor buffer (needs raw-frame handling).`);
      else if (same) out.push(`new VideoFrame(<video>) at θ ${theta}° matches the <video> (${sizes.join(', ')}): upright for the UI.`);
      if (theta === '0' || differs) break; // one line is enough
    }
  }
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
  /** Wait this long before recording (camera latency settles; default 0). */
  warmupMs?: number;
  /** Grabs per second, cycling through the GrabKinds (default 10, as FramePump). */
  grabHz?: number;
  /** maxSide of the downscaled createImageBitmap (default 1280, the 'fast' preset's). */
  scaledMaxSide?: number;
  createWorker?: () => ProbeWorker;
  /** Phase and remaining ms of it, at each grab. */
  onProgress?: (remainingMs: number, phase: 'warmup' | 'measure') => void;
  /** Stop early (e.g. the screen was left); the report covers the time measured. */
  isCancelled?: () => boolean;
}

type VideoWithRVFC = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (n: number, md: Record<string, number>) => void) => number;
  cancelVideoFrameCallback?: (h: number) => void;
};

/** One grab of `kind` from the <video>: main-thread (sync) time, time to result, result size. */
async function grabOnce(v: HTMLVideoElement, kind: GrabKind, scaledMaxSide: number): Promise<ProbeGrabSample | null> {
  const theta = screenAngle();
  const g0 = performance.now();
  if (kind === 'videoFrame') {
    const VF = (globalThis as { VideoFrame?: new (src: HTMLVideoElement) => { displayWidth: number; displayHeight: number; close(): void } }).VideoFrame;
    if (typeof VF !== 'function') return null;
    const f = new VF(v);
    const dt = performance.now() - g0;
    const sample: ProbeGrabSample = { kind, syncMs: dt, totalMs: dt, width: f.displayWidth, height: f.displayHeight, screenAngle: theta };
    f.close();
    return sample;
  }
  const w = v.videoWidth;
  const h = v.videoHeight;
  const s = kind === 'bitmapScaled' ? Math.min(1, scaledMaxSide / Math.max(w, h)) : 1;
  const p = s < 1 ? createImageBitmap(v, { resizeWidth: Math.round(w * s), resizeHeight: Math.round(h * s), resizeQuality: 'medium' }) : createImageBitmap(v);
  const g1 = performance.now();
  const bmp = await p;
  const sample: ProbeGrabSample = { kind, syncMs: g1 - g0, totalMs: performance.now() - g0, width: bmp.width, height: bmp.height, screenAngle: theta };
  bmp.close();
  return sample;
}

/**
 * Run the probe on a RUNNING CameraCapture (its <video> must be playing). The camera is left
 * running; the worker's track clone is stopped.
 */
export async function runCameraTimingProbe(camera: CameraCapture, opts: ProbeOptions = {}): Promise<CameraProbeReport> {
  const durationMs = opts.durationMs ?? 10_000;
  const warmupMs = opts.warmupMs ?? 0;
  const grabHz = opts.grabHz ?? 10;
  const track = camera.track;
  if (!track || track.readyState !== 'live') throw new Error('Camera timing probe: the camera is not running');

  // warm-up: the camera's delivery latency settles after it was opened
  const w0 = performance.now();
  while (performance.now() - w0 < warmupMs && !opts.isCancelled?.()) {
    opts.onProgress?.(Math.max(0, warmupMs - (performance.now() - w0)), 'warmup');
    await sleep(Math.min(250, Math.max(0, warmupMs - (performance.now() - w0))));
  }

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

  // main-thread side: rVFC samples and grabs (cycling through the kinds), for durationMs
  const v = camera.video as VideoWithRVFC | null;
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
  const kinds: GrabKind[] = ['bitmap', 'bitmapScaled', 'videoFrame'];
  let next = 0;
  const t0 = performance.now();
  const grabEvery = 1000 / grabHz;
  while (performance.now() - t0 < durationMs && !opts.isCancelled?.()) {
    if (v && v.readyState >= 2 && v.videoWidth) {
      try {
        const g = await grabOnce(v, kinds[next++ % kinds.length], opts.scaledMaxSide ?? 1280);
        if (g) grabs.push(g);
      } catch {
        /* not counted */
      }
    }
    opts.onProgress?.(Math.max(0, durationMs - (performance.now() - t0)), 'measure');
    await sleep(Math.max(0, grabEvery - ((performance.now() - t0) % grabEvery)));
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
  return analyzeCameraProbe({ durationMs: measuredMs, warmupMs, video, mstp, grabs, mstpError, mstpTransport: mstpError === null ? transport : null, grabHz });
}

function defaultProbeWorker(): ProbeWorker {
  return new Worker(new URL('../workers/camera-probe.worker.ts', import.meta.url), { type: 'module', name: 'camera-probe' }) as unknown as ProbeWorker;
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
