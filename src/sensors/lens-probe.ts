/**
 * Lens & photo probe (Diagnostics): measures, on the device, the two capture changes that are
 * worth building only if the browser cooperates (docs/research/ios-camera.md §2, §5.5, §8 items
 * 1, 6, 7):
 *
 *  1. Lenses: every camera enumerateDevices() lists, with its zoom range, minimum focus distance
 *     and largest size (InputDeviceInfo.getCapabilities()).
 *  2. Photos: on the running camera's track, `ImageCapture.takePhoto()` a few times (and once
 *     asking for the largest size getPhotoCapabilities() offers): latency, JPEG size, the size
 *     actually delivered, the EXIF focal length / 35 mm equivalent / exposure / ISO, and whether
 *     the video stalls or drops frames around each shot. One `grabFrame()` for comparison. A
 *     shutter sound cannot be detected from the page: the Diagnostics screen asks.
 *  3. Two cameras at once: open a second rear camera (ultra-wide first) while the first runs, and
 *     record whether both deliver frames, or the first is muted / ended, and whether it recovers
 *     when the second is closed.
 *  4. Per lens: open each physical rear camera in turn and take one photo: the EXIF focal length
 *     gives each lens's field of view (a prior for an ultra-wide option).
 *
 * `analyzeLensProbe` (pure) turns the raw samples into the report and its summary lines;
 * `parseJpeg` (pure) reads the JPEG size and the EXIF fields from the start of a photo.
 */
import { CameraCapture, classifyCamera, type CameraInfo, type LensClass } from './camera';

// ---------------------------------------------------------------------------------------------
// JPEG / EXIF
// ---------------------------------------------------------------------------------------------

export interface PhotoExif {
  make?: string;
  model?: string;
  lensModel?: string;
  orientation?: number;
  /** mm */
  focalLength?: number;
  focalLength35?: number;
  /** s */
  exposureTime?: number;
  iso?: number;
  pixelWidth?: number;
  pixelHeight?: number;
}

export interface JpegInfo {
  /** From the SOF marker (encoded size, before any EXIF orientation). */
  width: number | null;
  height: number | null;
  exif: PhotoExif | null;
}

/**
 * Size and EXIF of a JPEG from its first bytes (the SOF and APP1 come before the scan); also the
 * size of a PNG (Chromium's fake camera takes PNG photos).
 */
export function parseJpeg(b: Uint8Array): JpegInfo {
  const out: JpegInfo = { width: null, height: null, exif: null };
  if (b.length >= 24 && b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47) {
    const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
    out.width = dv.getUint32(16);
    out.height = dv.getUint32(20);
    return out;
  }
  if (b.length < 4 || b[0] !== 0xff || b[1] !== 0xd8) return out;
  let p = 2;
  while (p + 4 <= b.length) {
    if (b[p] !== 0xff) {
      p++;
      continue;
    }
    const m = b[p + 1];
    if (m === 0xff) {
      p++;
      continue;
    }
    if (m === 0xd9 || m === 0xda) break; // end of image / start of scan
    if (m === 0x01 || (m >= 0xd0 && m <= 0xd7)) {
      p += 2;
      continue;
    }
    const len = (b[p + 2] << 8) | b[p + 3];
    const s = p + 4;
    if (m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc && s + 5 <= b.length) {
      out.height = (b[s + 1] << 8) | b[s + 2];
      out.width = (b[s + 3] << 8) | b[s + 4];
    } else if (m === 0xe1 && !out.exif && s + 6 <= b.length && String.fromCharCode(b[s], b[s + 1], b[s + 2], b[s + 3]) === 'Exif') {
      out.exif = parseTiff(b.subarray(s + 6, Math.min(b.length, p + 2 + len)));
    }
    p += 2 + len;
  }
  return out;
}

const TYPE_SIZE: Record<number, number> = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8 };

function parseTiff(t: Uint8Array): PhotoExif | null {
  if (t.length < 8) return null;
  const le = t[0] === 0x49 && t[1] === 0x49;
  if (!le && !(t[0] === 0x4d && t[1] === 0x4d)) return null;
  const dv = new DataView(t.buffer, t.byteOffset, t.byteLength);
  const u16 = (o: number): number => dv.getUint16(o, le);
  const u32 = (o: number): number => dv.getUint32(o, le);
  const exif: PhotoExif = {};
  const readIfd = (off: number, onTag: (tag: number, read: () => number | string | null) => void): void => {
    if (off <= 0 || off + 2 > t.length) return;
    const n = u16(off);
    for (let i = 0; i < n; i++) {
      const e = off + 2 + i * 12;
      if (e + 12 > t.length) return;
      const tag = u16(e);
      const type = u16(e + 2);
      const count = u32(e + 4);
      const size = (TYPE_SIZE[type] ?? 0) * count;
      const at = size <= 4 ? e + 8 : u32(e + 8);
      onTag(tag, () => {
        if (!size || at + Math.min(size, 8) > t.length) return null;
        switch (type) {
          case 2: {
            let s = '';
            for (let k = 0; k < count && at + k < t.length && t[at + k]; k++) s += String.fromCharCode(t[at + k]);
            return s.trim();
          }
          case 1:
          case 7:
            return t[at];
          case 3:
            return u16(at);
          case 4:
            return u32(at);
          case 9:
            return dv.getInt32(at, le);
          case 5:
          case 10: {
            const num = type === 5 ? u32(at) : dv.getInt32(at, le);
            const den = type === 5 ? u32(at + 4) : dv.getInt32(at + 4, le);
            return den ? num / den : null;
          }
          default:
            return null;
        }
      });
    }
  };
  const num = (v: number | string | null): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);
  const str = (v: number | string | null): string | undefined => (typeof v === 'string' && v ? v : undefined);
  let exifIfd = 0;
  readIfd(u32(4), (tag, read) => {
    if (tag === 0x010f) exif.make = str(read());
    else if (tag === 0x0110) exif.model = str(read());
    else if (tag === 0x0112) exif.orientation = num(read());
    else if (tag === 0x8769) exifIfd = num(read()) ?? 0;
  });
  readIfd(exifIfd, (tag, read) => {
    if (tag === 0x829a) exif.exposureTime = num(read());
    else if (tag === 0x8827) exif.iso = num(read());
    else if (tag === 0x920a) exif.focalLength = num(read());
    else if (tag === 0xa405) exif.focalLength35 = num(read());
    else if (tag === 0xa434) exif.lensModel = str(read());
    else if (tag === 0xa002) exif.pixelWidth = num(read());
    else if (tag === 0xa003) exif.pixelHeight = num(read());
  });
  return exif;
}

/**
 * Horizontal (long-side) field of view from the 35 mm-equivalent focal length. The equivalent is
 * defined on the diagonal (43.27 mm), so the long side of a w×h image spans 43.27·w/√(w²+h²).
 */
export function hfovFrom35mm(focal35: number, w = 4, h = 3): number {
  const long = Math.max(w, h);
  const span = (43.27 * long) / Math.hypot(w, h);
  return (2 * Math.atan(span / (2 * focal35)) * 180) / Math.PI;
}

// ---------------------------------------------------------------------------------------------
// Analysis (pure)
// ---------------------------------------------------------------------------------------------

export interface LensInfo {
  label: string;
  lens: LensClass;
  facing: string;
  zoom: { min?: number; max?: number } | null;
  focusDistanceMinM: number | null;
  maxWidth: number | null;
  maxHeight: number | null;
  maxFrameRate: number | null;
}

export interface PhotoShot {
  /** What takePhoto() was asked for (null: default settings). */
  requested: { imageWidth: number; imageHeight: number } | null;
  /** performance.now() at the call and when the Blob arrived. */
  t0: number;
  t1: number;
  bytes: number;
  type: string;
  width: number | null;
  height: number | null;
  exif: PhotoExif | null;
  error?: string;
}

export interface TrackEvent {
  t: number;
  track: 'first' | 'second';
  type: string;
}

export interface LensPhoto {
  label: string;
  lens: LensClass;
  error: string | null;
  videoWidth: number | null;
  videoHeight: number | null;
  zoom: { min?: number; max?: number } | null;
  focusDistanceMinM: number | null;
  photo: { width: number | null; height: number | null; latencyMs: number; exif: PhotoExif | null } | null;
}

export interface LensProbeInput {
  lenses: LensInfo[];
  firstCamera: { label: string; lens: LensClass; width: number; height: number; frameRate: number };
  /** Callback times (ms) of every frame of the first camera, from the start to the end. */
  frames: number[];
  baseline: { t0: number; t1: number };
  photo: {
    available: boolean;
    error: string | null;
    capabilities: Record<string, unknown> | null;
    settings: Record<string, unknown> | null;
    shots: PhotoShot[];
    grabFrame: { ms: number; width: number; height: number } | { error: string } | null;
    trackBefore: Record<string, unknown> | null;
    trackAfter: Record<string, unknown> | null;
  };
  concurrent: {
    second: { label: string; lens: LensClass } | null;
    error: string | null;
    openMs: number | null;
    /** When the second track was open (performance.now()). */
    t0: number | null;
    t1: number | null;
    /** Callback times of the second camera's frames. */
    secondFrames: number[];
    events: TrackEvent[];
    /** First camera's track after the second was closed and a 2 s wait. */
    firstAfter: { readyState: string; muted: boolean } | null;
    /** Window after the second was closed, for the first camera's frame rate. */
    after: { t0: number; t1: number } | null;
  };
  perLens: LensPhoto[];
}

export interface Dist {
  n: number;
  p50: number;
  max: number;
}

export interface ShotReport {
  requested: string | null;
  latencyMs: number;
  bytes: number;
  type: string;
  size: string | null;
  focalLength: number | null;
  focalLength35: number | null;
  exposureTime: number | null;
  iso: number | null;
  /** Longest gap between first-camera frames from the call to 500 ms after the Blob. */
  videoMaxGapMs: number | null;
  videoDropped: number;
  error?: string;
}

export type ConcurrentVerdict = 'both-ran' | 'first-stopped' | 'second-failed' | 'second-no-frames' | 'not-run';

export interface LensProbeReport {
  lenses: LensInfo[];
  firstCamera: LensProbeInput['firstCamera'];
  photo: {
    available: boolean;
    error: string | null;
    capabilities: Record<string, unknown> | null;
    settings: Record<string, unknown> | null;
    videoFps: number | null;
    shots: ShotReport[];
    latencyMs: Dist | null;
    sizes: string[];
    /** The shot asking for the largest size came out larger than the default ones (null: not tried). */
    largerOnRequest: boolean | null;
    grabFrame: LensProbeInput['photo']['grabFrame'];
    /** The video track's size / frame rate changed after photos were taken. */
    trackChanged: string | null;
  };
  concurrent: {
    second: LensProbeInput['concurrent']['second'];
    verdict: ConcurrentVerdict;
    error: string | null;
    openMs: number | null;
    firstFpsBefore: number | null;
    firstFpsDuring: number | null;
    secondFps: number | null;
    firstFpsAfter: number | null;
    events: string[];
    firstAfter: LensProbeInput['concurrent']['firstAfter'];
  };
  perLens: (LensPhoto & { hfovDeg: number | null })[];
  summary: string[];
}

/** After the Blob arrives, how long the probe keeps watching the video for a stall. */
export const SHOT_TAIL_MS = 500;

const round = (x: number, d = 1): number => Math.round(x * 10 ** d) / 10 ** d;

function dist(values: number[]): Dist | null {
  if (!values.length) return null;
  const s = [...values].sort((a, b) => a - b);
  return { n: s.length, p50: round(s[(s.length - 1) >> 1]), max: round(s[s.length - 1]) };
}

function median(xs: number[]): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function fps(frames: number[], t0: number, t1: number): number | null {
  if (!(t1 > t0)) return null;
  const n = frames.filter((t) => t >= t0 && t < t1).length;
  return round((n * 1000) / (t1 - t0));
}

/**
 * Longest gap between consecutive frames across [a, b] (from the last frame before `a` to the
 * first after `b`; a missing end counts from / to the window's edge) and the frames missing there.
 */
export function stallIn(frames: readonly number[], a: number, b: number, periodMs: number): { maxGapMs: number; dropped: number } {
  let prev = a;
  let next = b;
  const inside: number[] = [];
  for (const t of frames) {
    if (t <= a) prev = t;
    else if (t <= b) inside.push(t);
    else {
      next = t;
      break;
    }
  }
  const pts = [prev, ...inside, next];
  let maxGap = 0;
  let dropped = 0;
  for (let i = 1; i < pts.length; i++) {
    const g = pts[i] - pts[i - 1];
    if (g > maxGap) maxGap = g;
    if (periodMs > 0 && g > 1.5 * periodMs) dropped += Math.round(g / periodMs) - 1;
  }
  return { maxGapMs: round(maxGap), dropped };
}

const sizeOf = (w: number | null | undefined, h: number | null | undefined): string | null => (w && h ? `${w}x${h}` : null);

export function analyzeLensProbe(inp: LensProbeInput): LensProbeReport {
  const { baseline, frames } = inp;
  const baseFrames = frames.filter((t) => t >= baseline.t0 && t < baseline.t1);
  const period = median(baseFrames.slice(1).map((t, i) => t - baseFrames[i])) ?? 1000 / (inp.firstCamera.frameRate || 30);

  // photos
  const ph = inp.photo;
  const shots: ShotReport[] = ph.shots.map((s) => {
    const st = s.error ? { maxGapMs: null, dropped: 0 } : stallIn(frames, s.t0, s.t1 + SHOT_TAIL_MS, period);
    return {
      requested: s.requested ? `${s.requested.imageWidth}x${s.requested.imageHeight}` : null,
      latencyMs: round(s.t1 - s.t0),
      bytes: s.bytes,
      type: s.type,
      size: sizeOf(s.width, s.height),
      focalLength: s.exif?.focalLength ?? null,
      focalLength35: s.exif?.focalLength35 ?? null,
      exposureTime: s.exif?.exposureTime ?? null,
      iso: s.exif?.iso ?? null,
      videoMaxGapMs: st.maxGapMs,
      videoDropped: st.dropped,
      ...(s.error ? { error: s.error } : {}),
    };
  });
  const ok = shots.filter((s) => !s.error);
  const plain = ok.filter((s) => !s.requested);
  const asked = ok.filter((s) => s.requested);
  const area = (s: ShotReport): number => (s.size ? s.size.split('x').map(Number).reduce((a, b) => a * b, 1) : 0);
  const largerOnRequest = asked.length && plain.length ? asked.some((s) => area(s) > Math.max(...plain.map(area))) : null;
  const tb = ph.trackBefore;
  const ta = ph.trackAfter;
  const trackKey = (s: Record<string, unknown> | null): string | null => (s ? `${s.width}x${s.height} @ ${s.frameRate} fps` : null);
  const trackChanged = tb && ta && trackKey(tb) !== trackKey(ta) ? `${trackKey(tb)} → ${trackKey(ta)}` : null;

  // two cameras
  const c = inp.concurrent;
  const firstFpsBefore = fps(frames, baseline.t0, baseline.t1);
  const firstFpsDuring = c.t0 !== null && c.t1 !== null ? fps(frames, c.t0 + 500, c.t1) : null; // skip the switch-over
  const secondFps = c.t0 !== null && c.t1 !== null ? fps(c.secondFrames, c.t0 + 500, c.t1) : null;
  const firstFpsAfter = c.after ? fps(frames, c.after.t0, c.after.t1) : null;
  const firstStopped = c.events.some((e) => e.track === 'first' && (e.type === 'ended' || e.type === 'mute')) || (firstFpsDuring !== null && firstFpsBefore !== null && firstFpsDuring < 0.25 * firstFpsBefore);
  let verdict: ConcurrentVerdict;
  if (!c.second) verdict = 'not-run';
  else if (c.error) verdict = 'second-failed';
  else if (!secondFps) verdict = firstStopped ? 'first-stopped' : 'second-no-frames';
  else verdict = firstStopped ? 'first-stopped' : 'both-ran';
  const t00 = c.t0 ?? 0;

  const perLens = inp.perLens.map((l) => {
    const f35 = l.photo?.exif?.focalLength35;
    const w = l.photo?.width ?? l.videoWidth ?? 4;
    const h = l.photo?.height ?? l.videoHeight ?? 3;
    return { ...l, hfovDeg: f35 ? round(hfovFrom35mm(f35, w, h)) : null };
  });

  const report: LensProbeReport = {
    lenses: inp.lenses,
    firstCamera: inp.firstCamera,
    photo: {
      available: ph.available,
      error: ph.error,
      capabilities: ph.capabilities,
      settings: ph.settings,
      videoFps: firstFpsBefore,
      shots,
      latencyMs: dist(plain.map((s) => s.latencyMs)),
      sizes: [...new Set(ok.map((s) => s.size).filter((s): s is string => !!s))],
      largerOnRequest,
      grabFrame: ph.grabFrame,
      trackChanged,
    },
    concurrent: {
      second: c.second,
      verdict,
      error: c.error,
      openMs: c.openMs === null ? null : round(c.openMs),
      firstFpsBefore,
      firstFpsDuring,
      secondFps,
      firstFpsAfter,
      events: c.events.map((e) => `${round(e.t - t00, 0)} ms ${e.track} ${e.type}`),
      firstAfter: c.firstAfter,
    },
    perLens,
    summary: [],
  };
  report.summary = summarizeLensProbe(report, period);
  return report;
}

function summarizeLensProbe(r: LensProbeReport, periodMs: number): string[] {
  const out: string[] = [];
  const back = r.lenses.filter((l) => l.facing === 'back');
  if (back.length) out.push(`Rear cameras: ${back.map((l) => `${l.label} (${l.lens})`).join(', ')}.`);

  const p = r.photo;
  if (!p.available) out.push(`takePhoto(): unavailable${p.error ? ` — ${p.error}` : ''}.`);
  else if (!p.shots.some((s) => !s.error)) out.push(`takePhoto(): every shot failed${p.shots[0]?.error ? ` (${p.shots[0].error})` : ''}.`);
  else {
    const ok = p.shots.filter((s) => !s.error);
    const v = r.firstCamera;
    const lat = p.latencyMs;
    out.push(
      `takePhoto(): ${p.sizes.join(', ') || 'size unknown'} JPEG (video ${v.width}x${v.height})` +
        (lat ? `, latency p50 ${lat.p50} ms / max ${lat.max} ms over ${lat.n} shot${lat.n === 1 ? '' : 's'}` : '') +
        `, ${round(ok.reduce((a, s) => a + s.bytes, 0) / ok.length / 1e6, 2)} MB each.`,
    );
    if (p.largerOnRequest !== null) out.push(p.largerOnRequest ? 'Asking for the largest size gave a LARGER photo.' : 'Asking for the largest size did not change the photo size (as expected, §5.5).');
    const f = ok.find((s) => s.focalLength !== null || s.focalLength35 !== null);
    out.push(f ? `EXIF: focal length ${f.focalLength ?? '?'} mm, ${f.focalLength35 ?? '?'} mm equivalent.` : 'EXIF: no focal length in the photos.');
    const gaps = ok.map((s) => s.videoMaxGapMs).filter((g): g is number => g !== null);
    const dropped = ok.reduce((a, s) => a + s.videoDropped, 0);
    const worst = gaps.length ? Math.max(...gaps) : null;
    if (worst !== null) {
      const stall = worst > 3 * periodMs;
      out.push(
        `Video during photos: worst frame gap ${worst} ms (normal ${round(periodMs)} ms), ${dropped} frame${dropped === 1 ? '' : 's'} dropped over ${ok.length} shot${ok.length === 1 ? '' : 's'} — ` +
          (stall ? 'the video STALLS while a photo is taken.' : 'the video keeps running.'),
      );
    }
    if (p.trackChanged) out.push(`The video track changed after the photos: ${p.trackChanged}.`);
  }
  const g = p.grabFrame;
  if (g && 'error' in g) out.push(`grabFrame(): failed — ${g.error}.`);
  else if (g) out.push(`grabFrame(): ${g.width}x${g.height} in ${round(g.ms)} ms (the video frame, not a photo).`);

  const c = r.concurrent;
  const name = c.second ? `${c.second.label} (${c.second.lens})` : '';
  switch (c.verdict) {
    case 'not-run':
      out.push('Two cameras at once: not tested (no second rear camera).');
      break;
    case 'second-failed':
      out.push(`Two cameras at once: opening ${name} failed (${c.error}).`);
      break;
    case 'first-stopped':
      out.push(
        `Two cameras at once: NO — opening ${name} stopped the first camera (${c.firstFpsDuring ?? 0} fps while both were open; second ${c.secondFps ?? 0} fps)` +
          (c.firstAfter ? `; after closing the second, the first was ${c.firstAfter.readyState}${c.firstAfter.muted ? ', muted' : ''} at ${c.firstFpsAfter ?? 0} fps.` : '.'),
      );
      break;
    case 'second-no-frames':
      out.push(`Two cameras at once: ${name} opened but delivered no frames while the first kept ${c.firstFpsDuring ?? '?'} fps.`);
      break;
    case 'both-ran':
      out.push(`Two cameras at once: YES — the first camera kept ${c.firstFpsDuring} fps (before: ${c.firstFpsBefore}) while ${name} ran at ${c.secondFps} fps.`);
      break;
  }

  for (const l of r.perLens) {
    if (l.error) out.push(`${l.label}: ${l.error}.`);
    else if (l.photo) {
      const e = l.photo.exif;
      out.push(
        `${l.label}: photo ${sizeOf(l.photo.width, l.photo.height) ?? '?'}, ${e?.focalLength ?? '?'} mm (${e?.focalLength35 ?? '?'} mm eq.)` +
          (l.hfovDeg !== null ? ` ≈ ${l.hfovDeg}° across the long side` : '') +
          (l.focusDistanceMinM !== null ? `, focuses from ${l.focusDistanceMinM} m` : '') +
          '.',
      );
    }
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Runner (browser)
// ---------------------------------------------------------------------------------------------

export type LensProbeStep = 'lenses' | 'photos' | 'concurrent' | 'perLens';

export interface LensProbeOptions {
  /** Photos with default settings (default 4), one second apart. */
  shots?: number;
  /** How long both cameras stay open (default 3000 ms). */
  concurrentMs?: number;
  /**
   * Where the second camera (and each lens in turn) plays. Keep it visible: WebKit may pause a
   * muted video that is off-screen. Default: a small preview fixed at the bottom right.
   */
  secondVideo?: HTMLVideoElement;
  onProgress?: (step: LensProbeStep, detail: string) => void;
  /** Stop early (e.g. the screen was left); the report covers what was measured. */
  isCancelled?: () => boolean;
}

interface ImageCaptureLike {
  takePhoto(settings?: Record<string, unknown>): Promise<Blob>;
  getPhotoCapabilities?(): Promise<unknown>;
  getPhotoSettings?(): Promise<unknown>;
  grabFrame?(): Promise<ImageBitmap>;
}
type ImageCaptureCtor = new (track: MediaStreamTrack) => ImageCaptureLike;

type VideoWithRVFC = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: () => void) => number;
  cancelVideoFrameCallback?: (h: number) => void;
};

/** The JPEG header bytes read for the size and EXIF (Apple's APP segments fit well within this). */
const HEADER_BYTES = 512 * 1024;
const PHOTO_TIMEOUT_MS = 6000;

function errText(e: unknown): string {
  const n = (e as { name?: string })?.name;
  const m = (e as Error)?.message;
  return n && m ? `${n}: ${m}` : (n ?? m ?? String(e));
}

function withTimeout<T>(p: Promise<T>, ms: number, what: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const id = setTimeout(() => reject(new Error(`${what} timed out after ${ms} ms`)), ms);
    p.then(
      (v) => {
        clearTimeout(id);
        resolve(v);
      },
      (e) => {
        clearTimeout(id);
        reject(e);
      },
    );
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

/** Plain copy of a WebIDL dictionary-like value (capabilities / settings) for the report. */
function plain(v: unknown): Record<string, unknown> | null {
  if (!v || typeof v !== 'object') return null;
  const out: Record<string, unknown> = {};
  for (const k in v as Record<string, unknown>) {
    const x = (v as Record<string, unknown>)[k];
    if (typeof x !== 'function') out[k] = x && typeof x === 'object' && !Array.isArray(x) ? plain(x) : x;
  }
  return out;
}

function range(caps: Record<string, unknown> | null | undefined, key: string): { min?: number; max?: number } | null {
  const v = caps?.[key] as { min?: number; max?: number } | undefined;
  return v && typeof v === 'object' && !Array.isArray(v) ? { min: v.min, max: v.max } : null;
}

function lensInfo(c: CameraInfo): LensInfo {
  const caps = c.capabilities;
  const fd = range(caps, 'focusDistance')?.min;
  return {
    label: c.label,
    lens: c.lens,
    facing: c.facing,
    zoom: range(caps, 'zoom'),
    focusDistanceMinM: typeof fd === 'number' && fd > 0 ? fd : null,
    maxWidth: range(caps, 'width')?.max ?? null,
    maxHeight: range(caps, 'height')?.max ?? null,
    maxFrameRate: range(caps, 'frameRate')?.max ?? null,
  };
}

async function shoot(ic: ImageCaptureLike, requested: PhotoShot['requested']): Promise<PhotoShot> {
  const t0 = performance.now();
  try {
    const blob = await withTimeout(ic.takePhoto(requested ?? undefined), PHOTO_TIMEOUT_MS, 'takePhoto()');
    const t1 = performance.now();
    const info = parseJpeg(new Uint8Array(await blob.slice(0, HEADER_BYTES).arrayBuffer()));
    return { requested, t0, t1, bytes: blob.size, type: blob.type, width: info.width, height: info.height, exif: info.exif };
  } catch (e) {
    return { requested, t0, t1: performance.now(), bytes: 0, type: '', width: null, height: null, exif: null, error: errText(e) };
  }
}

/** Count a <video>'s frames (callback times) until the returned stop() is called. */
function countFrames(v: VideoWithRVFC, into: number[]): () => void {
  let on = true;
  let h = 0;
  const cb = (): void => {
    if (!on) return;
    into.push(performance.now());
    h = v.requestVideoFrameCallback!(cb);
  };
  if (v.requestVideoFrameCallback) h = v.requestVideoFrameCallback(cb);
  return () => {
    on = false;
    v.cancelVideoFrameCallback?.(h);
  };
}

async function openStream(deviceId: string, size: { width: number; height: number }): Promise<MediaStream> {
  return withTimeout(
    navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: deviceId }, width: { ideal: size.width }, height: { ideal: size.height } }, audio: false }),
    6000,
    'getUserMedia()',
  );
}

async function playIn(v: HTMLVideoElement, stream: MediaStream): Promise<void> {
  v.srcObject = stream;
  try {
    await withTimeout(v.play(), 3000, 'play()');
  } catch {
    /* frames are counted either way */
  }
}

/**
 * Run the probe on a RUNNING CameraCapture started with `restartOnInterruption: false` (opening
 * a second camera may end its track, and a restart would then fight the probe). The camera is
 * stopped at the end: the per-lens step opens every rear camera in turn.
 */
export async function runLensProbe(camera: CameraCapture, opts: LensProbeOptions = {}): Promise<LensProbeReport> {
  const track = camera.track;
  const res = camera.result;
  if (!track || track.readyState !== 'live' || !res) throw new Error('Lens & photo probe: the camera is not running');
  const cancelled = (): boolean => !!opts.isCancelled?.();
  const progress = opts.onProgress ?? (() => {});

  // 1. lenses
  progress('lenses', 'Listing cameras');
  const cams = await CameraCapture.listCameras();
  const lenses = cams.map(lensInfo);

  const frames: number[] = [];
  const unFrame = camera.onFrame((_t, meta) => frames.push(meta.callbackTime));
  const events: TrackEvent[] = [];
  const watch = (t: MediaStreamTrack, which: TrackEvent['track']): (() => void) => {
    const types = ['mute', 'unmute', 'ended'] as const;
    const fns = types.map((type) => {
      const fn = (): void => void events.push({ t: performance.now(), track: which, type });
      t.addEventListener(type, fn);
      return fn;
    });
    return () => types.forEach((type, i) => t.removeEventListener(type, fns[i]));
  };
  const unwatchFirst = watch(track, 'first');

  const input: LensProbeInput = {
    lenses,
    firstCamera: { label: res.label, lens: res.lens, width: res.width, height: res.height, frameRate: res.frameRate },
    frames,
    baseline: { t0: 0, t1: 0 },
    photo: { available: false, error: null, capabilities: null, settings: null, shots: [], grabFrame: null, trackBefore: null, trackAfter: null },
    concurrent: { second: null, error: null, openMs: null, t0: null, t1: null, secondFrames: [], events, firstAfter: null, after: null },
    perLens: [],
  };
  const ownVideo = !opts.secondVideo;
  const video2 = (opts.secondVideo ?? document.createElement('video')) as VideoWithRVFC;
  video2.muted = true;
  video2.playsInline = true;
  video2.setAttribute('playsinline', '');
  if (ownVideo) {
    Object.assign(video2.style, { position: 'fixed', right: '8px', bottom: '8px', width: '96px', height: '72px', zIndex: '1000', background: '#000', pointerEvents: 'none' });
    document.body.appendChild(video2);
  }

  try {
    // baseline video rate
    progress('photos', 'Measuring the video');
    input.baseline.t0 = performance.now();
    await sleep(2000);
    input.baseline.t1 = performance.now();

    // 2. photos on the running track
    const IC = (globalThis as { ImageCapture?: ImageCaptureCtor }).ImageCapture;
    if (!cancelled() && typeof IC === 'function') {
      const ph = input.photo;
      ph.available = true;
      ph.trackBefore = plain(track.getSettings());
      try {
        const ic = new IC(track);
        try {
          ph.capabilities = plain(await withTimeout(ic.getPhotoCapabilities?.() ?? Promise.resolve(null), 3000, 'getPhotoCapabilities()'));
          ph.settings = plain(await withTimeout(ic.getPhotoSettings?.() ?? Promise.resolve(null), 3000, 'getPhotoSettings()'));
        } catch (e) {
          ph.error = errText(e);
        }
        const n = opts.shots ?? 4;
        for (let i = 0; i < n && !cancelled(); i++) {
          progress('photos', `Photo ${i + 1} of ${n + 1}`);
          ph.shots.push(await shoot(ic, null));
          await sleep(1000);
        }
        const wMax = range(ph.capabilities, 'imageWidth')?.max;
        const hMax = range(ph.capabilities, 'imageHeight')?.max;
        if (!cancelled() && wMax && hMax) {
          progress('photos', `Photo ${n + 1} of ${n + 1} (largest size)`);
          ph.shots.push(await shoot(ic, { imageWidth: wMax, imageHeight: hMax }));
          await sleep(1000);
        }
        if (!cancelled() && typeof ic.grabFrame === 'function') {
          const g0 = performance.now();
          try {
            const bmp = await withTimeout(ic.grabFrame(), 3000, 'grabFrame()');
            ph.grabFrame = { ms: performance.now() - g0, width: bmp.width, height: bmp.height };
            bmp.close();
          } catch (e) {
            ph.grabFrame = { error: errText(e) };
          }
        }
      } catch (e) {
        ph.error = errText(e);
      }
      ph.trackAfter = track.readyState === 'live' ? plain(track.getSettings()) : null;
    } else if (typeof IC !== 'function') input.photo.error = 'ImageCapture is not available';

    // 3. two cameras at once
    const firstId = res.deviceId;
    const order: LensClass[] = ['ultrawide', 'tele', 'wide', 'unknown', 'virtual'];
    const candidates = cams.filter((c) => c.facing === 'back' && c.deviceId && c.deviceId !== firstId && c.label !== res.label);
    candidates.sort((a, b) => order.indexOf(a.lens) - order.indexOf(b.lens));
    const second = candidates[0];
    if (!cancelled() && second && track.readyState === 'live') {
      const c = input.concurrent;
      c.second = { label: second.label, lens: second.lens };
      progress('concurrent', `Opening ${second.label} too`);
      const o0 = performance.now();
      let stream: MediaStream | null = null;
      try {
        stream = await openStream(second.deviceId, { width: 1280, height: 960 });
        c.openMs = performance.now() - o0;
      } catch (e) {
        c.error = errText(e);
      }
      if (stream) {
        const t2 = stream.getVideoTracks()[0];
        const unwatch2 = t2 ? watch(t2, 'second') : () => {};
        c.t0 = performance.now();
        await playIn(video2, stream);
        const stop2 = countFrames(video2, c.secondFrames);
        await sleep(opts.concurrentMs ?? 3000);
        c.t1 = performance.now();
        stop2();
        unwatch2();
        stream.getTracks().forEach((t) => t.stop());
        video2.srcObject = null;
        progress('concurrent', 'Closed the second camera');
        const a0 = performance.now();
        await sleep(2000);
        c.after = { t0: a0 + 500, t1: performance.now() };
        c.firstAfter = { readyState: track.readyState, muted: track.muted };
      }
    }
    // 4. one photo per physical rear lens
    camera.stop();
    if (typeof IC === 'function') {
      const physical = cams.filter((c) => c.facing === 'back' && c.deviceId && c.lens !== 'virtual');
      for (const cam of physical) {
        if (cancelled()) break;
        progress('perLens', cam.label || cam.lens);
        const lp: LensPhoto = { label: cam.label, lens: cam.lens, error: null, videoWidth: null, videoHeight: null, zoom: null, focusDistanceMinM: null, photo: null };
        input.perLens.push(lp);
        let stream: MediaStream | null = null;
        try {
          await sleep(300); // the previous camera finishes closing
          stream = await openStream(cam.deviceId, { width: 1920, height: 1440 });
          const t = stream.getVideoTracks()[0];
          await playIn(video2, stream);
          await sleep(800); // exposure / focus settle
          const s = t.getSettings();
          const caps = plain(typeof t.getCapabilities === 'function' ? t.getCapabilities() : null);
          const cls = classifyCamera(t.label || cam.label, caps ?? undefined);
          if (cls.lens !== 'unknown') lp.lens = cls.lens;
          lp.videoWidth = s.width ?? null;
          lp.videoHeight = s.height ?? null;
          lp.zoom = range(caps, 'zoom');
          const fd = range(caps, 'focusDistance')?.min;
          lp.focusDistanceMinM = typeof fd === 'number' && fd > 0 ? fd : null;
          const shot = await shoot(new IC(t), null);
          if (shot.error) lp.error = shot.error;
          else lp.photo = { width: shot.width, height: shot.height, latencyMs: round(shot.t1 - shot.t0), exif: shot.exif };
        } catch (e) {
          lp.error = errText(e);
        } finally {
          stream?.getTracks().forEach((t) => t.stop());
          video2.srcObject = null;
        }
      }
    }
  } finally {
    unwatchFirst();
    unFrame();
    video2.srcObject = null;
    if (ownVideo) video2.remove();
  }
  return analyzeLensProbe(input);
}
