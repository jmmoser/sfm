/**
 * Device / browser capability report for the diagnostics screen and bug reports. Every section
 * is collected defensively (a failing probe yields null + an entry in `errors`).
 *
 * `gpu: 'device'` (default) uses the shared `getGPU()` of src/gpu/device.ts, which creates the
 * WebGPU device of THIS realm (call it from the diagnostics screen, not during capture, when the
 * engine owns the device in its worker); `'adapter'` only requests an adapter.
 */
import { simdAvailable, simdStatus } from '../features/simd';
import { getGPU } from '../gpu/device';
import { CameraCapture, isAppleMobileWebKit, type CameraInfo } from './camera';
import { screenAngle } from './frame-orientation';
import type { GeoSensor } from './geolocation';
import { MotionSensor } from './motion';
import type { CompassSensor } from './orientation';
import { WakeLock } from './wakelock';

export interface GpuDiagnostics {
  available: boolean;
  source: 'device' | 'adapter';
  isSoftware?: boolean;
  info?: { vendor: string; architecture: string; description: string; device?: string };
  features?: string[];
  limits?: Record<string, number>;
  wgslLanguageFeatures?: string[];
  preferredCanvasFormat?: string;
}

export interface SensorDiagnostics {
  collectedAt: string;
  userAgent: string;
  platform: string;
  language: string;
  appleMobileWebKit: boolean;
  standalone: boolean;
  secureContext: boolean;
  crossOriginIsolated: boolean;
  hardwareConcurrency: number | null;
  deviceMemoryGB: number | null;
  devicePixelRatio: number;
  screen: { width: number; height: number; orientationType: string | null; orientationAngle: number | null; windowOrientation: number | null; screenAngle: number };
  features: Record<string, boolean>;
  gpu: GpuDiagnostics | null;
  cameras: CameraInfo[] | null;
  camera: { state: string; segment: number; result: unknown; settings: Record<string, unknown>; capabilities: Record<string, unknown> } | null;
  motion: { running: boolean; rateHz: number; calibration: unknown; stats: unknown } | null;
  compass: {
    running: boolean;
    yawOffsetDeg: number | null;
    yawAccuracyDeg: number | null;
    headingModel: string;
    /** 'true' once the WMM declination is applied (first GPS fix), else 'magnetic'. */
    northReference: string;
    declination: unknown;
    latest: unknown;
    stats: unknown;
  } | null;
  geo: { status: string; fixes: number; approximate: boolean; latest: unknown; lastError: string | null } | null;
  storage: { usage: number | null; quota: number | null; persisted: boolean | null } | null;
  /** WebGPU, camera and IndexedDB all missing — Lockdown Mode disables them. */
  lockdownSuspected: boolean;
  errors: string[];
}

export interface DiagnosticsOptions {
  camera?: CameraCapture | null;
  motion?: MotionSensor | null;
  compass?: CompassSensor | null;
  geo?: GeoSensor | null;
  gpu?: 'device' | 'adapter' | false;
  /** enumerateDevices (labels only after a camera grant). Default true. */
  listCameras?: boolean;
}

function limitsToPlain(l: GPUSupportedLimits | undefined): Record<string, number> {
  const out: Record<string, number> = {};
  if (!l) return out;
  for (const k in l) {
    const v = (l as unknown as Record<string, unknown>)[k];
    if (typeof v === 'number') out[k] = v;
  }
  return out;
}

/** Which of `names` exist in a dedicated worker; null when no worker could answer (2 s timeout). */
async function workerGlobals(names: string[], errors: string[]): Promise<Record<string, boolean> | null> {
  if (typeof Worker === 'undefined') return null;
  let w: Worker | null = null;
  try {
    w = new Worker(new URL('../workers/feature-probe.worker.ts', import.meta.url), { type: 'module', name: 'feature-probe' });
    const worker = w;
    return await new Promise<Record<string, boolean> | null>((resolve) => {
      const timer = setTimeout(() => resolve(null), 2000);
      worker.onmessage = (ev: MessageEvent<Record<string, boolean>>) => {
        clearTimeout(timer);
        resolve(ev.data);
      };
      worker.onerror = () => {
        clearTimeout(timer);
        resolve(null);
      };
      worker.postMessage(names);
    });
  } catch (e) {
    errors.push(`feature worker: ${(e as Error)?.message ?? e}`);
    return null;
  } finally {
    w?.terminate();
  }
}

export async function collectDiagnostics(opts: DiagnosticsOptions = {}): Promise<SensorDiagnostics> {
  const errors: string[] = [];
  const g = globalThis as unknown as Record<string, unknown> & {
    navigator?: Navigator & { deviceMemory?: number; standalone?: boolean; gpu?: GPU };
    screen?: Screen;
    matchMedia?: (q: string) => MediaQueryList;
  };
  const nav = g.navigator;
  const scr = g.screen;
  const has = (o: unknown, k: string): boolean => {
    try {
      return !!o && k in (o as object);
    } catch {
      return false;
    }
  };
  const standalone = (() => {
    try {
      return !!nav?.standalone || !!g.matchMedia?.('(display-mode: standalone)').matches;
    } catch {
      return false;
    }
  })();
  const features: Record<string, boolean> = {
    webgpu: !!nav?.gpu,
    getUserMedia: !!nav?.mediaDevices?.getUserMedia,
    requestVideoFrameCallback: typeof HTMLVideoElement !== 'undefined' && has(HTMLVideoElement.prototype, 'requestVideoFrameCallback'),
    // checked in a dedicated worker: Safari exposes MSTP there only (Chromium also on the main
    // thread). Only the camera timing probe uses it; the capture path grabs from the <video> on rVFC
    // (new VideoFrame on iOS by default, else createImageBitmap — src/sensors/frame-pump.ts).
    mediaStreamTrackProcessor: (await workerGlobals(['MediaStreamTrackProcessor'], errors))?.MediaStreamTrackProcessor ?? has(g, 'MediaStreamTrackProcessor'),
    videoFrame: has(g, 'VideoFrame'),
    imageCapture: has(g, 'ImageCapture'),
    offscreenCanvas: has(g, 'OffscreenCanvas'),
    createImageBitmap: typeof g.createImageBitmap === 'function',
    wakeLock: WakeLock.isSupported(),
    deviceMotion: MotionSensor.isSupported(),
    motionPermissionApi: MotionSensor.needsPermission(),
    deviceOrientation: has(g, 'DeviceOrientationEvent'),
    deviceOrientationAbsolute: has(g, 'ondeviceorientationabsolute'),
    geolocation: !!nav?.geolocation,
    sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    storageManager: !!nav?.storage,
    opfs: typeof nav?.storage?.getDirectory === 'function',
    indexedDB: has(g, 'indexedDB'),
    webShare: typeof nav?.share === 'function',
    screenOrientation: !!scr?.orientation,
    wasmSimd: (() => {
      try {
        // (module (func (result v128) i32.const 0 i8x16.splat i8x16.popcnt)) — wasm-feature-detect
        return WebAssembly.validate(
          new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]),
        );
      } catch {
        return false;
      }
    })(),
    // our SIMD kernels compiled AND passed the runtime known-answer self-test (Safari 26.x
    // v128 regression guard, features/simd.ts); false = plain-JS CPU kernels
    wasmSimdKernels: simdAvailable(),
  };
  const simd = simdStatus();
  if (simd.startsWith('self-test-failed')) errors.push(`wasm simd: ${simd}`);

  let gpu: GpuDiagnostics | null = null;
  const gpuMode = opts.gpu ?? 'device';
  if (gpuMode && nav?.gpu) {
    try {
      if (gpuMode === 'device') {
        const ctx = await getGPU();
        gpu = ctx
          ? {
              available: true,
              source: 'device',
              isSoftware: ctx.isSoftware,
              info: ctx.info,
              features: [...ctx.features].sort(),
              limits: limitsToPlain(ctx.limits),
            }
          : { available: false, source: 'device' };
      } else {
        const ad = await nav.gpu.requestAdapter();
        const info = (ad as GPUAdapter & { info?: GPUAdapterInfo } | null)?.info;
        gpu = ad
          ? {
              available: true,
              source: 'adapter',
              info: { vendor: info?.vendor ?? '', architecture: info?.architecture ?? '', description: info?.description ?? '', device: info?.device },
              features: [...(ad.features as unknown as Iterable<string>)].sort(),
              limits: limitsToPlain(ad.limits),
            }
          : { available: false, source: 'adapter' };
      }
      try {
        gpu.wgslLanguageFeatures = [...((nav.gpu as GPU & { wgslLanguageFeatures?: Iterable<string> }).wgslLanguageFeatures ?? [])].sort();
        gpu.preferredCanvasFormat = nav.gpu.getPreferredCanvasFormat();
      } catch {
        /* optional */
      }
    } catch (e) {
      errors.push(`gpu: ${(e as Error)?.message ?? e}`);
    }
  } else if (gpuMode) {
    gpu = { available: false, source: gpuMode };
  }

  let cameras: CameraInfo[] | null = null;
  if (opts.listCameras !== false) {
    try {
      cameras = await CameraCapture.listCameras();
    } catch (e) {
      errors.push(`cameras: ${(e as Error)?.message ?? e}`);
    }
  }

  let storage: SensorDiagnostics['storage'] = null;
  try {
    if (nav?.storage) {
      const est = nav.storage.estimate ? await nav.storage.estimate() : undefined;
      const persisted = nav.storage.persisted ? await nav.storage.persisted() : null;
      storage = { usage: est?.usage ?? null, quota: est?.quota ?? null, persisted };
    }
  } catch (e) {
    errors.push(`storage: ${(e as Error)?.message ?? e}`);
  }

  const cam = opts.camera;
  const mot = opts.motion;
  const cmp = opts.compass;
  const geo = opts.geo;
  const wo = (g as { orientation?: unknown }).orientation;
  return {
    collectedAt: new Date().toISOString(),
    userAgent: nav?.userAgent ?? '',
    platform: (nav as Navigator & { userAgentData?: { platform?: string } })?.userAgentData?.platform ?? nav?.platform ?? '',
    language: nav?.language ?? '',
    appleMobileWebKit: isAppleMobileWebKit(),
    standalone,
    secureContext: typeof isSecureContext === 'boolean' ? isSecureContext : false,
    crossOriginIsolated: typeof crossOriginIsolated === 'boolean' ? crossOriginIsolated : false,
    hardwareConcurrency: typeof nav?.hardwareConcurrency === 'number' ? nav.hardwareConcurrency : null,
    deviceMemoryGB: typeof nav?.deviceMemory === 'number' ? nav.deviceMemory : null,
    devicePixelRatio: typeof g.devicePixelRatio === 'number' ? (g.devicePixelRatio as number) : 1,
    screen: {
      width: scr?.width ?? 0,
      height: scr?.height ?? 0,
      orientationType: scr?.orientation?.type ?? null,
      orientationAngle: typeof scr?.orientation?.angle === 'number' ? scr.orientation.angle : null,
      windowOrientation: typeof wo === 'number' ? wo : null,
      screenAngle: screenAngle(),
    },
    features,
    gpu,
    cameras,
    camera: cam
      ? { state: cam.state, segment: cam.segment, result: cam.result, settings: cam.getSettings(), capabilities: cam.getCapabilities() }
      : null,
    motion: mot ? { running: mot.running, rateHz: mot.rateHz, calibration: { ...mot.calibration, gyroMap: Array.from(mot.calibration.gyroMap) }, stats: mot.stats } : null,
    compass: cmp
      ? {
          running: cmp.running,
          yawOffsetDeg: cmp.yawOffsetDeg,
          yawAccuracyDeg: cmp.yawAccuracyDeg,
          headingModel: cmp.headingModel,
          northReference: cmp.northReference,
          declination: cmp.declinationInfo,
          latest: cmp.latest(),
          stats: { ...cmp.stats },
        }
      : null,
    geo: geo ? { status: geo.status, fixes: geo.fixes.length, approximate: geo.approximate, latest: geo.latest(), lastError: geo.lastError } : null,
    storage,
    lockdownSuspected: !features.webgpu && !features.getUserMedia && !features.indexedDB,
    errors,
  };
}

/**
 * Keys that identify the device or the user's location. Not needed to debug the app, so never
 * part of a copied report: camera `deviceId` / `groupId` are stable per-origin identifiers, and
 * GPS fixes / WMM inputs (when a caller passes `geo` / `compass`) are the user's position.
 */
const PRIVATE_KEYS = new Set(['deviceId', 'groupId', 'lat', 'lon', 'alt', 'latDeg', 'lonDeg', 'altM', 'latitude', 'longitude', 'altitude', 'originLatLonAlt']);

/**
 * JSON text of a diagnostics report for the clipboard / bug reports: typed arrays become plain
 * arrays, and the values of PRIVATE_KEYS are replaced by `"[redacted]"` (empty values stay as
 * they are, so "no label / no id" remains visible).
 */
export function diagnosticsReportJSON(report: unknown): string {
  return JSON.stringify(
    report,
    (k, v: unknown) => {
      if (PRIVATE_KEYS.has(k) && v !== null && v !== undefined && v !== '') return '[redacted]';
      return ArrayBuffer.isView(v) && !(v instanceof DataView) ? Array.from(v as unknown as ArrayLike<number>) : v;
    },
    2,
  );
}
