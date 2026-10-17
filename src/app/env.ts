/**
 * Platform capability detection for the Home screen banners (docs/research/ios-platform.md
 * §6.5 Lockdown Mode, §6.4 HTTPS, ios-webgpu.md availability). `environmentIssues` is pure.
 */
import { isAppleMobileWebKit } from '../sensors/camera';

export type WebGPUState = 'available' | 'no-adapter' | 'unsupported' | 'unknown';

export interface EnvInfo {
  secureContext: boolean;
  /** iOS / iPadOS WebKit. */
  apple: boolean;
  /** Running as an installed Home Screen web app. */
  standalone: boolean;
  indexedDB: boolean;
  /** navigator.mediaDevices.getUserMedia exists. */
  camera: boolean;
  webgpu: WebGPUState;
  webgpuAdapter?: string;
  webgl: boolean;
  motion: boolean;
  share: boolean;
  /** WebGPU, camera and IndexedDB all missing — Lockdown Mode disables them together. */
  lockdownSuspected: boolean;
}

export interface EnvIssue {
  id: 'lockdown' | 'insecure' | 'camera' | 'storage' | 'webgl' | 'webgpu';
  severity: 'blocking' | 'warning';
  title: string;
  body: string;
}

export function detectEnvironment(): EnvInfo {
  const g = globalThis as unknown as {
    navigator?: Navigator & { standalone?: boolean; gpu?: unknown };
    indexedDB?: unknown;
    isSecureContext?: boolean;
    matchMedia?: (q: string) => MediaQueryList;
    DeviceMotionEvent?: unknown;
  };
  const nav = g.navigator;
  const standalone = (() => {
    try {
      return nav?.standalone === true || !!g.matchMedia?.('(display-mode: standalone)').matches;
    } catch {
      return false;
    }
  })();
  const webgpu: WebGPUState = nav?.gpu ? 'unknown' : 'unsupported';
  const camera = !!nav?.mediaDevices?.getUserMedia;
  let indexedDB = false;
  try {
    indexedDB = typeof g.indexedDB !== 'undefined' && g.indexedDB !== null;
  } catch {
    indexedDB = false;
  }
  return {
    secureContext: g.isSecureContext !== false,
    apple: isAppleMobileWebKit(),
    standalone,
    indexedDB,
    camera,
    webgpu,
    webgl: detectWebGL(),
    motion: typeof g.DeviceMotionEvent !== 'undefined',
    share: typeof nav?.share === 'function',
    lockdownSuspected: !nav?.gpu && !camera && !indexedDB,
  };
}

function detectWebGL(): boolean {
  try {
    if (typeof document === 'undefined') return false;
    const c = document.createElement('canvas');
    const gl = (c.getContext('webgl2') ?? c.getContext('webgl')) as WebGLRenderingContext | null;
    if (!gl) return false;
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return true;
  } catch {
    return false;
  }
}

/** Ask for a WebGPU adapter (cheap; no device). Resolves 'unknown' on timeout. */
export async function probeWebGPU(timeoutMs = 4000): Promise<{ state: WebGPUState; adapter?: string }> {
  const gpu = (globalThis as unknown as { navigator?: { gpu?: GPU } }).navigator?.gpu;
  if (!gpu) return { state: 'unsupported' };
  try {
    const ad = await Promise.race([gpu.requestAdapter(), new Promise<'timeout'>((r) => setTimeout(() => r('timeout'), timeoutMs))]);
    if (ad === 'timeout') return { state: 'unknown' };
    if (!ad) return { state: 'no-adapter' };
    const info = (ad as GPUAdapter & { info?: GPUAdapterInfo }).info;
    const name = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(' ').trim();
    return { state: 'available', adapter: name || 'WebGPU adapter' };
  } catch {
    return { state: 'no-adapter' };
  }
}

/** Blocking problems first, then warnings. Lockdown Mode replaces the individual issues. */
export function environmentIssues(env: EnvInfo): EnvIssue[] {
  const out: EnvIssue[] = [];
  if (env.lockdownSuspected) {
    out.push({
      id: 'lockdown',
      severity: 'blocking',
      title: 'Lockdown Mode is blocking this app',
      body: 'Lockdown Mode turns off the camera, on-device storage and GPU features that scanning needs. To use the app, exclude this website: Settings › Privacy & Security › Lockdown Mode › Configure Web Browsing.',
    });
    return out;
  }
  if (!env.secureContext) {
    out.push({
      id: 'insecure',
      severity: 'blocking',
      title: 'Secure connection required',
      body: 'The camera and motion sensors only work over HTTPS. Open the app from an https:// address.',
    });
  }
  if (!env.camera && env.secureContext) {
    out.push({
      id: 'camera',
      severity: 'blocking',
      title: 'Camera not available',
      body: 'This browser does not allow camera access here. Try Safari (or Chrome) and make sure camera access is not disabled for websites.',
    });
  }
  if (!env.indexedDB) {
    out.push({
      id: 'storage',
      severity: 'blocking',
      title: 'Storage not available',
      body: 'Scans are saved on this device with IndexedDB, which is turned off here (Private Browsing or a privacy setting). Open the app in a normal window.',
    });
  }
  if (!env.webgl) {
    out.push({
      id: 'webgl',
      severity: 'warning',
      title: '3D preview unavailable',
      body: 'WebGL is turned off, so models cannot be displayed. Capture and export still work.',
    });
  }
  if (env.webgpu === 'unsupported' || env.webgpu === 'no-adapter') {
    out.push({
      id: 'webgpu',
      severity: 'warning',
      title: 'CPU mode — slower',
      body: 'WebGPU is not available on this device or browser. Scanning works, but processing takes several times longer. iOS 26 or later is recommended.',
    });
  }
  return out;
}

/** Blocking issues prevent starting a scan. */
export function canScan(env: EnvInfo): boolean {
  return !environmentIssues(env).some((i) => i.severity === 'blocking');
}
