/**
 * Camera capture for photogrammetry (getUserMedia → <video> → requestVideoFrameCallback).
 *
 * iOS facts this module is built on (docs/research/ios-camera.md):
 *  - `facingMode: 'environment'` picks a VIRTUAL multi-lens camera (Triple / Dual Wide) that
 *    switches lenses (and therefore intrinsics) on its own. We select the physical
 *    "Back Camera" (fixed wide-angle lens) by deviceId. Labels are localised and only exist
 *    after a camera permission grant (we probe once if needed).
 *  - Formats = the camera's native AVFoundation formats; sizes are matched in SENSOR-LANDSCAPE
 *    numbers whatever the UI orientation (always request landscape). No constraints ⇒ 640×480.
 *    4:3 ≥ 1920×1440 keep the full sensor FOV (≈ 67° on a 26 mm main camera); 16:9 crops
 *    vertically; 1440×1080 and ≤ 1024×768 are cropped. There is no native 1280×960, so the
 *    'fast' preset uses 1920×1440 on Apple devices and suggests a 1280 px grab size instead.
 *  - Settable: zoom (kept at 1), torch, whiteBalanceMode, frameRate. NOT settable on iOS:
 *    focusMode/focusDistance/exposure (lockFocus() therefore returns false there; it works on
 *    Chrome/Android). Video stabilisation is off; OIS cannot be disabled.
 *  - Frames from <video>/createImageBitmap are upright for the current UI orientation; the
 *    rotation is metadata applied on display (see frame-orientation.ts for R_cam_device).
 *  - rVFC `captureTime` = when the frame reached WebKit's capture source (after ISP, ~10–40 ms
 *    after exposure; same performance.now() timeline); `mediaTime` is 0 for camera streams.
 *  - Hidden page ⇒ the track is muted (not ended) and unmuted on return. If it is still muted
 *    ~1 s after becoming visible, or ended, we re-open the same device (new `segment`).
 */
export type LensClass = 'wide' | 'ultrawide' | 'tele' | 'virtual' | 'front' | 'unknown';
export type CameraFacing = 'back' | 'front' | 'unknown';
export type CaptureQuality = 'fast' | 'balanced' | 'max';
export type CameraState = 'idle' | 'starting' | 'running' | 'interrupted' | 'restarting' | 'stopped' | 'error';

export interface CameraInfo {
  deviceId: string;
  label: string;
  lens: LensClass;
  facing: CameraFacing;
  groupId?: string;
  /** `InputDeviceInfo.getCapabilities()` when available (Safari 17+, Chrome). */
  capabilities?: Record<string, unknown>;
}

export interface FrameMeta {
  /** rVFC mediaTime (s); 0 for camera streams on iOS. */
  mediaTime: number;
  /** rVFC captureTime (ms, performance.now() domain) when provided and plausible. */
  captureTime?: number;
  presentationTime?: number;
  expectedDisplayTime?: number;
  presentedFrames?: number;
  /** Frame size as delivered (rotated for the UI orientation on iOS). */
  width: number;
  height: number;
  /** performance.now() when the callback ran. */
  callbackTime: number;
  /** Which value the `t` argument of the frame callback was taken from. */
  timeSource: 'captureTime' | 'presentationTime' | 'callback';
  /** Increments whenever capture resumed after an interruption (start a new SfM segment). */
  segment: number;
}

/** `t` = best capture-time estimate (ms, performance.now() domain). `meta` is reused: copy what you keep. */
export type FrameCallback = (t: number, meta: FrameMeta) => void;

export interface CameraStartOptions {
  quality: CaptureQuality;
  deviceId?: string;
  /** Used when no deviceId is given (default 'back'). */
  facing?: 'back' | 'front';
  /** Ideal frame rate (default 30). */
  frameRate?: number;
  /** Explicit landscape size to try first (exact, then the preset list). */
  size?: { width: number; height: number };
  /** Re-open the camera after 'ended' or a stuck interruption (default true). */
  restartOnInterruption?: boolean;
  /** Stop the camera while the page is hidden and re-open on return (default false). */
  releaseWhenHidden?: boolean;
}

export interface CameraStartResult {
  /** Delivered frame size (UI-rotated on iOS: e.g. 1440×1920 in a portrait UI). */
  width: number;
  height: number;
  frameRate: number;
  label: string;
  deviceId: string;
  /**
   * Prior of the field of view across the LONG image side (sensor-landscape horizontal FOV),
   * degrees. Use {@link focalPriorPx} (= (max(W,H)/2) / tan(hfov/2)) for the focal length so
   * the value is valid for any UI rotation. ±10 %.
   */
  hfovPriorDeg: number;
  /** Focal length prior in pixels of the delivered frames. */
  focalPriorPx: number;
  capabilities: Record<string, unknown>;
  settings: Record<string, unknown>;
  lens: LensClass;
  facing: CameraFacing;
  aspect: '4:3' | '16:9' | 'other';
  /** True for virtual multi-lens devices or when the physical lens could not be identified. */
  lensMaySwitch: boolean;
  /** Apple (iOS/iPadOS WebKit) camera stack detected. */
  apple: boolean;
  /** Suggested `grabFrame({maxSide})` for the quality preset (null = native size). */
  frameMaxSide: number | null;
  /** Landscape size that was requested for the successful stream (null = fallback). */
  requested: { width: number; height: number } | null;
  /** Log of constraint attempts (for diagnostics). */
  attempts: string[];
  /** Minimum focus distance (m) from capabilities.focusDistance.min, if reported. */
  minFocusDistanceM: number | null;
  torchAvailable: boolean;
}

// ---------------------------------------------------------------------------------------------
// Label classification (pure)
// ---------------------------------------------------------------------------------------------

// English AVFoundation names: "Back Camera", "Back Ultra Wide Camera", "Back Telephoto Camera",
// "Back Dual Wide Camera", "Back Dual Camera", "Back Triple Camera", "Front Camera"; plus
// localised keywords (de, fr, es, it, pt, nl, pl, tr, ru, ja, zh, ko — best effort) and
// Android Chrome labels ("camera2 0, facing back").
const RE_BACK = /back|rear|rück|arrière|trasera|traseira|posterior|achter|tylny|arka|背面|后置|後置|후면|задн|environment/i;
const RE_FRONT = /front|vorder|avant|frontal|anterior|voorkant|przedni|ön kamera|前面|前置|전면|передн|facetime/i;
const RE_VIRTUAL =
  /triple|tripl[ae]|dreifach|drievoud|potrójn|üçlü|トリプル|三镜头|三鏡頭|트리플|тройн|dual|double|doble|dupla|duplo|doppia|doppel|dubbel|podwójn|çift|デュアル|双|雙|듀얼|двойн/i;
const RE_ULTRA = /ultra|超広角|超广角|超廣角|초광각|сверхширок|ультраширок/i;
const RE_TELE = /tele(?!camera)|téléobj|望遠|长焦|長焦|망원|телеоб/i;
const RE_APPLE_EN = /^(Back|Front) (Camera|Ultra Wide Camera|Telephoto Camera|Dual Wide Camera|Dual Camera|Triple Camera|TrueDepth Camera)$/;

/** Classify a camera from its (possibly localised) label. */
export function classifyCameraLabel(label: string): { lens: LensClass; facing: CameraFacing } {
  const l = (label || '').normalize('NFC');
  if (!l) return { lens: 'unknown', facing: 'unknown' };
  const back = RE_BACK.test(l);
  if (!back && RE_FRONT.test(l)) return { lens: 'front', facing: 'front' };
  if (RE_VIRTUAL.test(l)) return { lens: 'virtual', facing: 'back' };
  if (RE_ULTRA.test(l)) return { lens: 'ultrawide', facing: 'back' };
  if (RE_TELE.test(l)) return { lens: 'tele', facing: 'back' };
  if (back) return { lens: 'wide', facing: 'back' };
  return { lens: 'unknown', facing: 'unknown' };
}

function capRange(caps: Record<string, unknown> | undefined, key: string): { min?: number; max?: number } | null {
  const v = caps?.[key] as { min?: number; max?: number } | undefined;
  return v && typeof v === 'object' && !Array.isArray(v) ? v : null;
}
function capList(caps: Record<string, unknown> | undefined, key: string): string[] {
  const v = caps?.[key];
  return Array.isArray(v) ? (v as string[]) : [];
}

/** Label classification refined by device capabilities (facingMode, zoom.min < 1 ⇒ virtual). */
export function classifyCamera(label: string, caps?: Record<string, unknown>): { lens: LensClass; facing: CameraFacing } {
  const c = classifyCameraLabel(label);
  const fm = capList(caps, 'facingMode');
  let facing = c.facing;
  if (fm.includes('environment')) facing = 'back';
  else if (fm.includes('user')) facing = 'front';
  let lens = c.lens;
  if (facing === 'front') lens = 'front';
  const zoom = capRange(caps, 'zoom');
  // WebKit normalises Triple / Dual Wide zoom so 1 = wide and min = 0.5 (ultra-wide).
  if (lens !== 'front' && zoom && typeof zoom.min === 'number' && zoom.min < 0.99) lens = 'virtual';
  return { lens, facing };
}

/**
 * Pick the camera for photogrammetry: the fixed wide-angle rear lens; then an unidentified
 * rear camera (Android lists the main camera first); then a virtual multi-lens device (zoom
 * 1, may switch lenses); then unknown (desktop webcams); then ultra-wide, telephoto, front.
 */
export function chooseCamera(cams: readonly CameraInfo[], facing: 'back' | 'front' = 'back'): CameraInfo | null {
  if (!cams.length) return null;
  const rank = (c: CameraInfo): number => {
    if (facing === 'front') return c.lens === 'front' ? 0 : c.facing === 'front' ? 1 : c.lens === 'unknown' ? 2 : 5;
    switch (c.lens) {
      case 'wide':
        return 0;
      case 'unknown':
        return c.facing === 'back' ? 1 : 3;
      case 'virtual':
        return 2;
      case 'ultrawide':
        return 4;
      case 'tele':
        return 5;
      default:
        return 9;
    }
  };
  let best = cams[0];
  let br = rank(best);
  for (const c of cams) {
    const r = rank(c);
    if (r < br) {
      best = c;
      br = r;
    }
  }
  // Several unidentified rear cameras (labels in a locale the keyword lists do not cover): iOS
  // enumerates Ultra Wide BEFORE the wide camera, so "first" would pick the ultra-wide. The
  // ultra-wide has the shortest minimum focus distance and the telephoto the longest
  // (docs/research/ios-camera.md §2 step 3, UNVERIFIED heuristic) ⇒ take the median.
  if (facing === 'back' && br === 1) {
    const tied = cams.filter((c) => rank(c) === 1);
    const fd = tied.map((c) => capRange(c.capabilities, 'focusDistance')?.min);
    if (tied.length > 1 && fd.every((v) => typeof v === 'number' && v > 0)) {
      const order = tied.map((_, i) => i).sort((a, b) => (fd[a] as number) - (fd[b] as number) || a - b);
      return tied[order[tied.length >> 1]];
    }
  }
  return best;
}

/** True when the rear camera had to be picked among several unidentified ones (see chooseCamera). */
function ambiguousRearChoice(cams: readonly CameraInfo[]): boolean {
  return cams.filter((c) => c.lens === 'unknown' && c.facing === 'back').length > 1;
}

/** iOS/iPadOS WebKit (feature-detected first; UA only as a supporting signal). */
export function isAppleMobileWebKit(): boolean {
  const g = globalThis as unknown as {
    DeviceMotionEvent?: { requestPermission?: unknown };
    navigator?: { userAgent?: string; maxTouchPoints?: number };
  };
  if (typeof g.DeviceMotionEvent?.requestPermission === 'function') return true;
  const ua = g.navigator?.userAgent ?? '';
  return /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && (g.navigator?.maxTouchPoints ?? 0) > 1);
}

/**
 * FOV prior (degrees) across the LONG side of a frame of size width×height.
 * Apple main camera: 26 mm-eq ⇒ 67.3° (measured 67.10° on the 12 Pro Max), Pro models 24 mm
 * (14 Pro+) ⇒ 71.6°, 26 mm (11–13 Pro) ⇒ 67.3°: 69.5° is used when a telephoto/triple camera
 * exists. 16:9 formats: ×68.16/67.10 (measured). Cropped low-resolution 4:3 formats (Apple):
 * 1440×1080-class ×59.31/67.10, ≤ 1024×768 ×64.72/67.10. Ultra-wide ≈ 104° after distortion
 * correction, telephoto ≈ 25° (77 mm), front ≈ 70°, unknown (webcams) 63°.
 */
export function hfovPriorDeg(lens: LensClass, width: number, height: number, opts: { apple?: boolean; pro?: boolean } = {}): number {
  const long = Math.max(width, height);
  const short = Math.max(1, Math.min(width, height));
  const aspect = long / short;
  const is169 = Math.abs(aspect - 16 / 9) < 0.03;
  const is43 = Math.abs(aspect - 4 / 3) < 0.03;
  const apple = !!opts.apple;
  let base: number;
  switch (lens) {
    case 'wide':
    case 'virtual':
      base = apple ? (opts.pro ? 69.5 : 67.3) : 68;
      break;
    case 'ultrawide':
      base = apple ? 104 : 100;
      break;
    case 'tele':
      base = apple ? 25 : 30;
      break;
    case 'front':
      base = 70;
      break;
    default:
      base = 63;
  }
  if (apple && (lens === 'wide' || lens === 'virtual')) {
    if (is169) base *= 68.16 / 67.1;
    else if (is43 && long < 1900) base *= long <= 1100 ? 64.72 / 67.1 : 59.31 / 67.1;
  }
  return base;
}

/** Focal length (px) from a long-side FOV prior: (max(W,H)/2) / tan(hfov/2). */
export function focalPriorPx(hfovLongSideDeg: number, width: number, height: number): number {
  return Math.max(width, height) / 2 / Math.tan((hfovLongSideDeg * Math.PI) / 360);
}

const SIZES_43_MAX: readonly [number, number][] = [
  [4032, 3024],
  [3264, 2448],
  [2592, 1944],
  [1920, 1440],
];
const SIZES_169_MAX: readonly [number, number][] = [
  [3840, 2160],
  [1920, 1080],
];

/** Landscape sizes to try (exact) for a quality preset, filtered by the device's max size. */
export function candidateSizes(
  quality: CaptureQuality,
  apple: boolean,
  maxSize?: { width?: number; height?: number } | null,
): [number, number][] {
  let list: [number, number][];
  if (quality === 'max') list = [...SIZES_43_MAX, ...SIZES_169_MAX].map((s) => [s[0], s[1]]);
  else if (quality === 'balanced')
    list = apple
      ? [
          [1920, 1440],
          [1920, 1080],
        ]
      : [
          [1920, 1440],
          [1600, 1200],
          [1920, 1080],
        ];
  else
    list = apple
      ? [
          [1920, 1440],
          [1920, 1080],
        ]
      : [
          [1280, 960],
          [1280, 720],
        ];
  const mw = maxSize?.width;
  const mh = maxSize?.height;
  if (typeof mw === 'number' && typeof mh === 'number' && mw > 0 && mh > 0) {
    const ml = Math.max(mw, mh);
    const ms = Math.min(mw, mh);
    const f = list.filter(([w, h]) => w <= ml && h <= ms);
    if (f.length) list = f;
  }
  return list;
}

function toPlain(o: unknown): Record<string, unknown> {
  try {
    return JSON.parse(JSON.stringify(o ?? {})) as Record<string, unknown>;
  } catch {
    return {};
  }
}

function errName(e: unknown): string {
  return (e as { name?: string })?.name ?? 'Error';
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

type VideoWithRVFC = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number, md: Record<string, number>) => void) => number;
  cancelVideoFrameCallback?: (h: number) => void;
};

// ---------------------------------------------------------------------------------------------
// CameraCapture
// ---------------------------------------------------------------------------------------------

export class CameraCapture {
  /** Video inputs with lens classification. Labels are empty until camera permission was granted. */
  static async listCameras(): Promise<CameraInfo[]> {
    const md = typeof navigator !== 'undefined' ? navigator.mediaDevices : undefined;
    if (!md?.enumerateDevices) return [];
    const devs = await md.enumerateDevices();
    const out: CameraInfo[] = [];
    for (const d of devs) {
      if (d.kind !== 'videoinput') continue;
      let caps: Record<string, unknown> | undefined;
      try {
        const gc = (d as InputDeviceInfo).getCapabilities;
        if (typeof gc === 'function') caps = toPlain(gc.call(d));
      } catch {
        caps = undefined;
      }
      const cls = classifyCamera(d.label, caps);
      out.push({ deviceId: d.deviceId, label: d.label, lens: cls.lens, facing: cls.facing, groupId: d.groupId, capabilities: caps });
    }
    return out;
  }

  /** Pure chooser (see {@link chooseCamera}). */
  static chooseCamera(cams: readonly CameraInfo[], facing: 'back' | 'front' = 'back'): CameraInfo | null {
    return chooseCamera(cams, facing);
  }

  private videoEl: VideoWithRVFC | null = null;
  private mediaStream: MediaStream | null = null;
  private videoTrack: MediaStreamTrack | null = null;
  private constraints: MediaTrackConstraints | null = null;
  private applied: Record<string, unknown> = {};
  private startOpts: CameraStartOptions | null = null;
  private lastResult: CameraStartResult | null = null;
  private st: CameraState = 'idle';
  private seg = 0;
  private stopping = false;
  private startGen = 0;
  private restarting = false;
  private released = false;
  private restartTimes: number[] = [];
  private visTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly stateCbs = new Set<(s: CameraState, detail?: string) => void>();
  private readonly frameCbs = new Set<FrameCallback>();
  private loopHandle = 0;
  private loopKind: 'rvfc' | 'raf' | null = null;
  private lastRafFrames = -1;
  private lastRafTime = -1;
  private readonly meta: FrameMeta = {
    mediaTime: 0,
    width: 0,
    height: 0,
    callbackTime: 0,
    timeSource: 'callback',
    segment: 0,
  };

  get state(): CameraState {
    return this.st;
  }
  /** Capture segment: increments whenever frames resume after an interruption. */
  get segment(): number {
    return this.seg;
  }
  get video(): HTMLVideoElement | null {
    return this.videoEl;
  }
  get stream(): MediaStream | null {
    return this.mediaStream;
  }
  get track(): MediaStreamTrack | null {
    return this.videoTrack;
  }
  /** Result of the last successful start() (updated after restarts). */
  get result(): CameraStartResult | null {
    return this.lastResult;
  }

  onStateChange(cb: (s: CameraState, detail?: string) => void): () => void {
    this.stateCbs.add(cb);
    return () => this.stateCbs.delete(cb);
  }

  private setState(s: CameraState, detail?: string): void {
    if (s === this.st && !detail) return;
    this.st = s;
    for (const cb of this.stateCbs) {
      try {
        cb(s, detail);
      } catch (e) {
        console.error(e);
      }
    }
  }

  /**
   * Open the camera and attach it to `video` (made muted/playsinline/autoplay for iOS).
   * Call from a user gesture when possible (iOS reuses a camera grant without a gesture only
   * within ~1 min of the previous capture).
   */
  async start(video: HTMLVideoElement, opts: CameraStartOptions): Promise<CameraStartResult> {
    if (this.mediaStream) this.stop();
    const md = typeof navigator !== 'undefined' ? navigator.mediaDevices : undefined;
    if (!md?.getUserMedia) throw new Error('CameraCapture: getUserMedia is not available (HTTPS required)');
    const gen = ++this.startGen; // a later start()/stop() supersedes this one
    this.stopping = false;
    this.released = false;
    this.applied = {};
    this.startOpts = opts;
    this.videoEl = video as VideoWithRVFC;
    this.setState('starting');
    const attempts: string[] = [];
    try {
      prepareVideoElement(video);
      const facing = opts.facing ?? 'back';
      // 1. Device selection (labels need a grant: probe once if they are missing).
      let cams = await CameraCapture.listCameras();
      if (!opts.deviceId && cams.length > 0 && cams.every((c) => !c.label)) {
        try {
          const probe = await md.getUserMedia({ video: { facingMode: { ideal: facing === 'front' ? 'user' : 'environment' } }, audio: false });
          probe.getTracks().forEach((t) => t.stop());
          attempts.push('probe: ok');
        } catch (e) {
          attempts.push(`probe: ${errName(e)}`);
          if (errName(e) === 'NotAllowedError' || errName(e) === 'SecurityError') throw e;
        }
        cams = await CameraCapture.listCameras();
      }
      const chosen = opts.deviceId ? (cams.find((c) => c.deviceId === opts.deviceId) ?? null) : chooseCamera(cams, facing);
      const deviceId = opts.deviceId ?? (chosen?.deviceId || undefined);
      const apple = isAppleMobileWebKit() || cams.some((c) => RE_APPLE_EN.test(c.label));
      const pro = cams.some((c) => c.lens === 'tele' || /triple/i.test(c.label));
      const devCaps = chosen?.capabilities;
      const wr = capRange(devCaps, 'width');
      const hr = capRange(devCaps, 'height');
      const maxSize = wr?.max && hr?.max ? { width: wr.max, height: hr.max } : null;
      const fps = opts.frameRate ?? 30;
      const supported = md.getSupportedConstraints?.() ?? {};
      const resizeOk = !!(supported as Record<string, boolean>).resizeMode;

      // 2. Format selection: exact landscape sizes, then ideal, then anything.
      const sizes = candidateSizes(opts.quality, apple, maxSize);
      if (opts.size) sizes.unshift([Math.max(opts.size.width, opts.size.height), Math.min(opts.size.width, opts.size.height)]);
      const base = (): Record<string, unknown> => {
        const c: Record<string, unknown> = { frameRate: { ideal: fps } };
        if (deviceId) c.deviceId = { exact: deviceId };
        else c.facingMode = { ideal: facing === 'front' ? 'user' : 'environment' };
        return c;
      };
      const tries: { c: MediaTrackConstraints; req: { width: number; height: number } | null; desc: string }[] = [];
      for (const [w, h] of sizes) {
        const c = base();
        c.width = { exact: w };
        c.height = { exact: h };
        if (resizeOk) c.resizeMode = { exact: 'none' };
        tries.push({ c: c as MediaTrackConstraints, req: { width: w, height: h }, desc: `exact ${w}x${h}` });
      }
      {
        const [w, h] = sizes[0] ?? [1920, 1440];
        const c = base();
        c.width = { ideal: w };
        c.height = { ideal: h };
        if (resizeOk) c.resizeMode = { ideal: 'none' };
        tries.push({ c: c as MediaTrackConstraints, req: null, desc: `ideal ${w}x${h}` });
      }
      tries.push({ c: base() as MediaTrackConstraints, req: null, desc: 'device only' });
      let stream: MediaStream | null = null;
      let used: (typeof tries)[number] | null = null;
      let lastErr: unknown = null;
      for (const tr of tries) {
        try {
          stream = await md.getUserMedia({ video: tr.c, audio: false });
          used = tr;
          attempts.push(`${tr.desc}: ok`);
          break;
        } catch (e) {
          lastErr = e;
          const n = errName(e);
          attempts.push(`${tr.desc}: ${n}`);
          if (n === 'NotAllowedError' || n === 'SecurityError') throw e;
          if (n === 'NotReadableError' || n === 'AbortError') {
            await sleep(300); // camera busy (previous session still closing): one retry
            try {
              stream = await md.getUserMedia({ video: tr.c, audio: false });
              used = tr;
              attempts.push(`${tr.desc} (retry): ok`);
              break;
            } catch (e2) {
              lastErr = e2;
              attempts.push(`${tr.desc} (retry): ${errName(e2)}`);
              if (errName(e2) === 'NotReadableError') throw e2;
            }
          }
        }
      }
      if (!stream || !used) throw lastErr ?? new Error('CameraCapture: no camera stream');
      if (this.stopping || gen !== this.startGen) {
        stream.getTracks().forEach((t) => t.stop());
        throw new Error('CameraCapture: start() was superseded by stop() or another start()');
      }
      this.constraints = used.c;
      await this.attachStream(stream, gen);
      if (gen !== this.startGen) throw new Error('CameraCapture: start() was superseded by stop() or another start()');
      await this.applyInitialSettings();
      if (gen !== this.startGen) throw new Error('CameraCapture: start() was superseded by stop() or another start()');

      // 3. Result.
      const track = this.videoTrack!;
      const settings = toPlain(track.getSettings());
      const capabilities = toPlain(typeof track.getCapabilities === 'function' ? track.getCapabilities() : {});
      const vw = video.videoWidth || (settings.width as number) || 0;
      const vh = video.videoHeight || (settings.height as number) || 0;
      const label = track.label || chosen?.label || '';
      const cls = classifyCamera(label, { ...(devCaps ?? {}), ...capabilities });
      const lens = cls.lens === 'unknown' && chosen && chosen.lens !== 'unknown' ? chosen.lens : cls.lens;
      const facingRes = cls.facing !== 'unknown' ? cls.facing : (chosen?.facing ?? 'unknown');
      const long = Math.max(vw, vh);
      const short = Math.max(1, Math.min(vw, vh));
      const ar = long / short;
      const aspect = Math.abs(ar - 4 / 3) < 0.03 ? '4:3' : Math.abs(ar - 16 / 9) < 0.03 ? '16:9' : 'other';
      // An unidentified Apple rear camera is (after chooseCamera's tie-break) most likely the main
      // wide lens: use its prior (67–72°) rather than the generic webcam one (63°).
      const fovLens: LensClass = apple && lens === 'unknown' && facingRes === 'back' ? 'wide' : lens;
      const hfov = hfovPriorDeg(fovLens, vw, vh, { apple, pro });
      const zoomCap = capRange(capabilities, 'zoom');
      const fd = capRange(capabilities, 'focusDistance');
      const result: CameraStartResult = {
        width: vw,
        height: vh,
        frameRate: (settings.frameRate as number) ?? fps,
        label,
        deviceId: (settings.deviceId as string) ?? deviceId ?? '',
        hfovPriorDeg: hfov,
        focalPriorPx: focalPriorPx(hfov, vw, vh),
        capabilities,
        settings,
        lens,
        facing: facingRes,
        aspect,
        lensMaySwitch:
          lens === 'virtual' ||
          (!deviceId && apple) ||
          (lens === 'unknown' && !!zoomCap && (zoomCap.min ?? 1) < 0.99) ||
          // unknown locale on a multi-camera iPhone: the "Back Dual Camera" cannot be told apart
          (apple && lens === 'unknown' && !opts.deviceId && ambiguousRearChoice(cams)),
        apple,
        frameMaxSide: opts.quality === 'fast' && long > 1400 ? 1280 : null,
        requested: used.req,
        attempts,
        minFocusDistanceM: typeof fd?.min === 'number' && fd.min > 0 && fd.min < 5 ? fd.min : null,
        torchAvailable: capabilities.torch === true || (Array.isArray(capabilities.torch) && (capabilities.torch as unknown[]).includes(true)),
      };
      this.lastResult = result;
      this.addVisibilityListener();
      this.setState('running');
      this.ensureLoop();
      return result;
    } catch (e) {
      if (gen === this.startGen) {
        this.setState('error', errName(e));
        this.releaseStream();
      }
      throw e;
    }
  }

  /** Attach `stream` to the video element; returns early (no wait) once `gen` is superseded. */
  private async attachStream(stream: MediaStream, gen: number): Promise<void> {
    const video = this.videoEl!;
    this.mediaStream = stream;
    const track = stream.getVideoTracks()[0];
    if (!track) throw new Error('CameraCapture: stream has no video track');
    this.videoTrack = track;
    track.addEventListener('ended', this.onTrackEnded);
    track.addEventListener('mute', this.onTrackMute);
    track.addEventListener('unmute', this.onTrackUnmute);
    video.srcObject = stream;
    try {
      await Promise.race([video.play(), sleep(4000)]);
    } catch (e) {
      if (errName(e) === 'NotAllowedError') throw new Error('CameraCapture: video autoplay was blocked — call start() from a user gesture');
      if (errName(e) !== 'AbortError') throw e;
    }
    if (gen !== this.startGen || this.stopping) return; // stop()/start() meanwhile: caller bails out
    await waitForDimensions(video, 5000);
  }

  /** WebKit pauses the media elements of a backgrounded page; frames (rVFC) need it playing. */
  private resumeVideo(): void {
    const v = this.videoEl;
    if (v && this.mediaStream && v.paused && !this.stopping) {
      try {
        v.play().catch(() => undefined);
      } catch {
        /* ignore */
      }
    }
  }

  /**
   * Apply image settings (zoom, torch, focus/exposure/white-balance modes). applyConstraints()
   * REPLACES the whole constraint set, so the stream's size/rate/device constraints and all
   * previously applied settings are re-sent (each as its own advanced set: an unsatisfiable one
   * is skipped instead of voiding the others). Advanced sets fail silently, so the result is
   * verified against getSettings().
   */
  private async applyOne(c: Record<string, unknown>): Promise<boolean> {
    const track = this.videoTrack;
    if (!track) return false;
    const sets: Record<string, unknown>[] = [];
    for (const [k, v] of Object.entries(this.applied)) if (!(k in c)) sets.push({ [k]: v });
    sets.push(c);
    try {
      await track.applyConstraints({ ...(this.constraints ?? {}), advanced: sets } as unknown as MediaTrackConstraints);
    } catch {
      return false;
    }
    const st = this.settingsRaw();
    for (const [k, v] of Object.entries(c)) {
      if (!(k in st)) continue;
      const cur = st[k];
      const ok = typeof v === 'number' && typeof cur === 'number' ? Math.abs(cur - v) <= 0.02 * Math.max(1, Math.abs(v)) : cur === v;
      if (!ok) return false;
    }
    Object.assign(this.applied, c);
    return true;
  }

  /** Re-send all applied image settings (after a restart). */
  private async reapply(): Promise<void> {
    const track = this.videoTrack;
    if (!track || Object.keys(this.applied).length === 0) return;
    const sets = Object.entries(this.applied).map(([k, v]) => ({ [k]: v }));
    try {
      await track.applyConstraints({ ...(this.constraints ?? {}), advanced: sets } as unknown as MediaTrackConstraints);
    } catch {
      /* best effort */
    }
  }

  private caps(): Record<string, unknown> {
    const t = this.videoTrack;
    try {
      return t && typeof t.getCapabilities === 'function' ? (t.getCapabilities() as Record<string, unknown>) : {};
    } catch {
      return {};
    }
  }

  private settingsRaw(): Record<string, unknown> {
    try {
      return (this.videoTrack?.getSettings() as Record<string, unknown>) ?? {};
    } catch {
      return {};
    }
  }

  /** zoom = 1, continuous focus / exposure / white balance where supported, torch off. */
  private async applyInitialSettings(): Promise<void> {
    const caps = this.caps();
    const st = this.settingsRaw();
    const zoom = capRange(caps, 'zoom');
    if (zoom && typeof zoom.min === 'number' && typeof zoom.max === 'number' && zoom.min <= 1 && zoom.max >= 1 && st.zoom !== 1) {
      await this.applyOne({ zoom: 1 });
    }
    if (capList(caps, 'focusMode').includes('continuous') && st.focusMode !== 'continuous') await this.applyOne({ focusMode: 'continuous' });
    if (capList(caps, 'exposureMode').includes('continuous') && st.exposureMode !== 'continuous') await this.applyOne({ exposureMode: 'continuous' });
    if (capList(caps, 'whiteBalanceMode').includes('continuous') && st.whiteBalanceMode !== 'continuous')
      await this.applyOne({ whiteBalanceMode: 'continuous' });
    if (st.torch === true) await this.applyOne({ torch: false });
  }

  /** Stop capture and release the camera (frame subscriptions are kept for the next start). */
  stop(): void {
    this.stopping = true;
    this.startGen++;
    this.cancelLoop();
    this.removeVisibilityListener();
    this.releaseStream();
    if (this.videoEl) {
      try {
        this.videoEl.pause();
      } catch {
        /* ignore */
      }
      this.videoEl.srcObject = null;
    }
    this.setState('stopped');
  }

  private releaseStream(): void {
    const t = this.videoTrack;
    if (t) {
      t.removeEventListener('ended', this.onTrackEnded);
      t.removeEventListener('mute', this.onTrackMute);
      t.removeEventListener('unmute', this.onTrackUnmute);
    }
    this.mediaStream?.getTracks().forEach((tr) => tr.stop());
    this.mediaStream = null;
    this.videoTrack = null;
  }

  // ---- interruption handling ----------------------------------------------------------------

  private readonly onTrackEnded = (): void => {
    if (this.stopping) return;
    const hidden = typeof document !== 'undefined' && document.hidden;
    if (!hidden && (this.startOpts?.restartOnInterruption ?? true)) void this.restart('ended');
    else this.setState('interrupted', 'ended');
  };
  private readonly onTrackMute = (): void => {
    if (!this.stopping && this.st === 'running') this.setState('interrupted', 'muted');
  };
  private readonly onTrackUnmute = (): void => {
    if (this.stopping) return;
    if (this.st === 'interrupted') {
      this.seg++;
      this.setState('running', 'unmuted');
    }
    this.resumeVideo();
  };
  private readonly onVisibility = (): void => {
    if (this.stopping || typeof document === 'undefined') return;
    if (this.visTimer) {
      clearTimeout(this.visTimer);
      this.visTimer = null;
    }
    if (document.hidden) {
      if (this.startOpts?.releaseWhenHidden) {
        this.released = true;
        this.cancelLoop();
        this.releaseStream();
      }
      this.setState('interrupted', 'hidden');
      return;
    }
    if (this.released) {
      void this.restart('visible');
      return;
    }
    const cur = this.videoTrack;
    if (this.st === 'interrupted' && cur && cur.readyState === 'live' && !cur.muted) {
      // Track was not muted (desktop browsers, quick app switch): frames resume right away and
      // must already carry the new segment (the phone may have moved while hidden).
      this.seg++;
      this.setState('running', 'visible');
      this.resumeVideo();
    }
    // Give iOS ~1.2 s to unmute the track; re-open if it is stuck or ended.
    this.visTimer = setTimeout(() => {
      this.visTimer = null;
      const t = this.videoTrack;
      if (!t || t.readyState === 'ended' || t.muted) {
        if (this.startOpts?.restartOnInterruption ?? true) void this.restart('stuck');
      } else {
        if (this.st === 'interrupted') {
          this.seg++;
          this.setState('running', 'visible');
        }
        this.resumeVideo();
      }
    }, 1200);
  };

  private visListening = false;
  private addVisibilityListener(): void {
    if (this.visListening || typeof document === 'undefined') return;
    document.addEventListener('visibilitychange', this.onVisibility);
    this.visListening = true;
  }
  private removeVisibilityListener(): void {
    if (!this.visListening || typeof document === 'undefined') return;
    document.removeEventListener('visibilitychange', this.onVisibility);
    this.visListening = false;
    if (this.visTimer) clearTimeout(this.visTimer);
    this.visTimer = null;
  }

  /**
   * Re-open the same device with the constraints that worked (new capture segment). Automatic
   * restarts ('ended', 'stuck', …) are limited to 3 per 10 s (restart loops); 'manual' and
   * 'visible' ones are not. A concurrent stop()/start() supersedes the restart (its stream is
   * stopped, the state is left to the newer call). On iOS call it from a tap when possible: a
   * getUserMedia without a gesture reuses the grant only within ~1 min of capture stopping.
   */
  async restart(reason = 'manual'): Promise<boolean> {
    if (this.restarting || this.stopping || !this.videoEl || !this.constraints) return false;
    if (reason !== 'manual' && reason !== 'visible') {
      const now = Date.now();
      this.restartTimes = this.restartTimes.filter((t) => now - t < 10000);
      if (this.restartTimes.length >= 3) {
        this.setState('error', `restart limit (${reason})`);
        return false;
      }
      this.restartTimes.push(now);
    }
    const gen = this.startGen;
    const superseded = (): boolean => this.stopping || gen !== this.startGen;
    let stream: MediaStream | null = null;
    this.restarting = true;
    this.setState('restarting', reason);
    try {
      this.releaseStream();
      stream = await navigator.mediaDevices.getUserMedia({ video: this.constraints, audio: false });
      if (superseded()) {
        stream.getTracks().forEach((t) => t.stop());
        return false;
      }
      await this.attachStream(stream, gen);
      if (superseded()) {
        stream.getTracks().forEach((t) => t.stop()); // stop()/start() already released it; idempotent
        return false;
      }
      if (Object.keys(this.applied).length) await this.reapply();
      else await this.applyInitialSettings();
      if (superseded()) {
        stream.getTracks().forEach((t) => t.stop());
        return false;
      }
      this.released = false;
      this.seg++;
      if (this.lastResult) {
        this.lastResult.settings = toPlain(this.videoTrack?.getSettings());
        this.lastResult.width = this.videoEl.videoWidth || this.lastResult.width;
        this.lastResult.height = this.videoEl.videoHeight || this.lastResult.height;
      }
      this.setState('running', `restarted (${reason})`);
      this.ensureLoop();
      return true;
    } catch (e) {
      if (superseded()) {
        stream?.getTracks().forEach((t) => t.stop());
        return false;
      }
      this.releaseStream(); // do not keep a half-attached camera open (battery, privacy indicator)
      this.setState('error', `restart failed: ${errName(e)}`);
      return false;
    } finally {
      this.restarting = false;
    }
  }

  // ---- frames -------------------------------------------------------------------------------

  /**
   * Subscribe to presented video frames (requestVideoFrameCallback; rAF fallback that skips
   * repeated frames). `t` = best capture-time estimate: rVFC captureTime when plausible, else
   * presentationTime, else the callback time. Returns an unsubscribe function.
   */
  onFrame(cb: FrameCallback): () => void {
    this.frameCbs.add(cb);
    this.ensureLoop();
    return () => {
      this.frameCbs.delete(cb);
      if (this.frameCbs.size === 0) this.cancelLoop();
    };
  }

  private ensureLoop(): void {
    const v = this.videoEl;
    if (!v || this.loopKind || this.frameCbs.size === 0 || !this.mediaStream) return;
    if (typeof v.requestVideoFrameCallback === 'function') {
      this.loopKind = 'rvfc';
      this.loopHandle = v.requestVideoFrameCallback(this.onVideoFrame);
    } else if (typeof requestAnimationFrame === 'function') {
      this.loopKind = 'raf';
      this.loopHandle = requestAnimationFrame(this.onAnimationFrame);
    }
  }

  private cancelLoop(): void {
    const v = this.videoEl;
    if (this.loopKind === 'rvfc' && v && typeof v.cancelVideoFrameCallback === 'function') v.cancelVideoFrameCallback(this.loopHandle);
    else if (this.loopKind === 'raf' && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(this.loopHandle);
    this.loopKind = null;
    this.loopHandle = 0;
  }

  private readonly onVideoFrame = (now: number, md: Record<string, number>): void => {
    const v = this.videoEl;
    if (this.loopKind !== 'rvfc' || !v) return;
    if (this.frameCbs.size === 0 || !this.mediaStream) {
      this.loopKind = null;
      return;
    }
    this.loopHandle = v.requestVideoFrameCallback!(this.onVideoFrame); // re-arm first
    const m = this.meta;
    const cbTime = performance.now();
    m.mediaTime = md.mediaTime ?? 0;
    m.presentationTime = md.presentationTime;
    m.expectedDisplayTime = md.expectedDisplayTime;
    m.presentedFrames = md.presentedFrames;
    m.width = md.width || v.videoWidth;
    m.height = md.height || v.videoHeight;
    m.callbackTime = cbTime;
    m.segment = this.seg;
    const ct = md.captureTime;
    const plausible = (x: number | undefined): x is number => typeof x === 'number' && Number.isFinite(x) && x <= cbTime + 5 && x >= cbTime - 2000;
    let t: number;
    if (plausible(ct)) {
      m.captureTime = ct;
      m.timeSource = 'captureTime';
      t = ct;
    } else {
      m.captureTime = undefined;
      if (plausible(md.presentationTime)) {
        m.timeSource = 'presentationTime';
        t = md.presentationTime;
      } else {
        m.timeSource = 'callback';
        t = Number.isFinite(now) ? Math.min(now, cbTime) : cbTime;
      }
    }
    this.dispatch(t);
  };

  private readonly onAnimationFrame = (now: number): void => {
    const v = this.videoEl;
    if (this.loopKind !== 'raf' || !v) return;
    if (this.frameCbs.size === 0 || !this.mediaStream) {
      this.loopKind = null;
      return;
    }
    this.loopHandle = requestAnimationFrame(this.onAnimationFrame);
    if (v.readyState < 2 || !v.videoWidth) return;
    const q = typeof v.getVideoPlaybackQuality === 'function' ? v.getVideoPlaybackQuality().totalVideoFrames : -1;
    if (q > 0) {
      if (q === this.lastRafFrames) return;
      this.lastRafFrames = q;
    } else if (v.currentTime > 0) {
      if (v.currentTime === this.lastRafTime) return;
      this.lastRafTime = v.currentTime;
    }
    const m = this.meta;
    m.mediaTime = v.currentTime;
    m.captureTime = undefined;
    m.presentationTime = undefined;
    m.expectedDisplayTime = undefined;
    m.presentedFrames = q > 0 ? q : undefined;
    m.width = v.videoWidth;
    m.height = v.videoHeight;
    m.callbackTime = performance.now();
    m.timeSource = 'callback';
    m.segment = this.seg;
    this.dispatch(Number.isFinite(now) ? now : m.callbackTime);
  };

  private dispatch(t: number): void {
    for (const cb of this.frameCbs) {
      try {
        cb(t, this.meta);
      } catch (e) {
        console.error('[sensors] frame callback failed', e);
      }
    }
  }

  /**
   * The current video frame as an ImageBitmap (native resolution, upright for the current UI
   * orientation), optionally downscaled so that max(W,H) ≤ maxSide. The caller owns (and must
   * close or transfer) the bitmap. When the browser ignores the resize options the native size
   * is returned — always use bitmap.width/height.
   */
  async grabFrame(opts: { maxSide?: number; resizeQuality?: ResizeQuality } = {}): Promise<ImageBitmap> {
    const v = this.videoEl;
    if (!v || !this.mediaStream || v.readyState < 2 || !v.videoWidth || !v.videoHeight) {
      throw new Error('CameraCapture.grabFrame: no video frame available');
    }
    const w = v.videoWidth;
    const h = v.videoHeight;
    const ms = opts.maxSide;
    if (ms && ms > 0 && Math.max(w, h) > ms) {
      const s = ms / Math.max(w, h);
      try {
        return await createImageBitmap(v, {
          resizeWidth: Math.max(1, Math.round(w * s)),
          resizeHeight: Math.max(1, Math.round(h * s)),
          resizeQuality: opts.resizeQuality ?? 'medium',
        });
      } catch {
        /* fall through to native size */
      }
    }
    return createImageBitmap(v);
  }

  /**
   * The current video frame as a VideoFrame (synchronous, ~0 ms: a reference to the camera buffer,
   * no pixel copy). On iOS it keeps the SENSOR orientation (landscape in every UI rotation —
   * Diagnostics camera timing probe). The caller owns it and must close it promptly (it holds a
   * capture buffer). Throws without a frame or VideoFrame support.
   */
  grabVideoFrame(): VideoFrame {
    const v = this.videoEl;
    const VF = (globalThis as { VideoFrame?: new (src: HTMLVideoElement) => VideoFrame }).VideoFrame;
    if (typeof VF !== 'function') throw new Error('CameraCapture.grabVideoFrame: VideoFrame is not supported');
    if (!v || !this.mediaStream || v.readyState < 2 || !v.videoWidth || !v.videoHeight) {
      throw new Error('CameraCapture.grabVideoFrame: no video frame available');
    }
    return new VF(v);
  }

  // ---- controls -----------------------------------------------------------------------------

  /** Current track settings (plain object). */
  getSettings(): Record<string, unknown> {
    return toPlain(this.settingsRaw());
  }
  /** Current track capabilities (plain object). */
  getCapabilities(): Record<string, unknown> {
    return toPlain(this.caps());
  }

  async setTorch(on: boolean): Promise<boolean> {
    const caps = this.caps();
    const t = caps.torch;
    if (!(t === true || (Array.isArray(t) && t.includes(true)))) return false;
    const ok = await this.applyOne({ torch: on });
    if (!ok) return false;
    const st = this.settingsRaw();
    return st.torch === undefined || st.torch === on;
  }

  /** Set the zoom factor (clamped to the capability range). Any value ≠ 1 changes the intrinsics. */
  async setZoom(z: number): Promise<boolean> {
    const r = capRange(this.caps(), 'zoom');
    if (!r || typeof r.min !== 'number' || typeof r.max !== 'number') return false;
    return this.applyOne({ zoom: Math.min(r.max, Math.max(r.min, z)) });
  }

  /**
   * Lock focus at the current distance (manual focusMode + current focusDistance, else
   * single-shot). Returns false where focus is not controllable (iOS Safari).
   */
  async lockFocus(): Promise<boolean> {
    const caps = this.caps();
    const modes = capList(caps, 'focusMode');
    const st = this.settingsRaw();
    if (modes.includes('manual') && typeof st.focusDistance === 'number' && capRange(caps, 'focusDistance')) {
      if (await this.applyOne({ focusMode: 'manual', focusDistance: st.focusDistance })) return true;
    }
    if (modes.includes('manual') && (await this.applyOne({ focusMode: 'manual' }))) return true;
    if (modes.includes('single-shot')) return this.applyOne({ focusMode: 'single-shot' });
    return false;
  }

  async unlockFocus(): Promise<boolean> {
    return capList(this.caps(), 'focusMode').includes('continuous') ? this.applyOne({ focusMode: 'continuous' }) : false;
  }

  /** Lock white balance (iOS: AVCaptureWhiteBalanceModeLocked) — do it ~1 s after start. */
  async lockWhiteBalance(): Promise<boolean> {
    return capList(this.caps(), 'whiteBalanceMode').includes('manual') ? this.applyOne({ whiteBalanceMode: 'manual' }) : false;
  }

  async unlockWhiteBalance(): Promise<boolean> {
    return capList(this.caps(), 'whiteBalanceMode').includes('continuous') ? this.applyOne({ whiteBalanceMode: 'continuous' }) : false;
  }

  /** Lock exposure at the current exposure time (not supported on iOS). */
  async lockExposure(): Promise<boolean> {
    const caps = this.caps();
    if (!capList(caps, 'exposureMode').includes('manual')) return false;
    const st = this.settingsRaw();
    const c: Record<string, unknown> = { exposureMode: 'manual' };
    if (typeof st.exposureTime === 'number' && capRange(caps, 'exposureTime')) c.exposureTime = st.exposureTime;
    return this.applyOne(c);
  }

  async unlockExposure(): Promise<boolean> {
    return capList(this.caps(), 'exposureMode').includes('continuous') ? this.applyOne({ exposureMode: 'continuous' }) : false;
  }
}

function prepareVideoElement(video: HTMLVideoElement): void {
  // iOS: inline playback without user interaction requires muted + playsinline.
  video.muted = true;
  video.defaultMuted = true;
  video.playsInline = true;
  video.autoplay = true;
  video.setAttribute('muted', '');
  video.setAttribute('playsinline', '');
  video.setAttribute('webkit-playsinline', '');
  video.setAttribute('autoplay', '');
  video.disablePictureInPicture = true;
}

function waitForDimensions(video: HTMLVideoElement, timeoutMs: number): Promise<void> {
  if (video.readyState >= 2 && video.videoWidth > 0) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const done = (): void => {
      if (video.videoWidth > 0 && video.readyState >= 2) {
        cleanup();
        resolve();
      }
    };
    const timer = setTimeout(() => {
      cleanup();
      if (video.videoWidth > 0) resolve();
      else reject(new Error('CameraCapture: timed out waiting for the first video frame'));
    }, timeoutMs);
    const cleanup = (): void => {
      clearTimeout(timer);
      video.removeEventListener('loadeddata', done);
      video.removeEventListener('loadedmetadata', done);
      video.removeEventListener('resize', done);
      video.removeEventListener('canplay', done);
    };
    video.addEventListener('loadeddata', done);
    video.addEventListener('loadedmetadata', done);
    video.addEventListener('resize', done);
    video.addEventListener('canplay', done);
  });
}
