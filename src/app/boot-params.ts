/**
 * URL parameters read at boot (src/main.ts), validated: anything unknown or malformed is ignored.
 *
 *   mock=1 | true          in-page MockEngine (demo / tests)
 *   mockSpeed=<x>          MockEngine time scale (0.05 … 50)
 *   mockFail=<method>[:message]   a MockEngine method fails (init | frame | refine | densify | mesh |
 *                          export | loadSession). The custom message is a DEVELOPMENT-build test hook
 *                          only: production builds show a fixed text, so a crafted link cannot put
 *                          arbitrary words (fake warnings, phone numbers…) into the app's own UI.
 *   mockStall=<method>[,…] a long MockEngine operation hangs (finishCapture | refine | densify | mesh)
 *   mockMemoryLimit=<n>    capture memory cap after n keyframes (integer ≥ 1)
 *   screen=diagnostics|setup   open that screen first
 *   camera=mstp           capture frames with MediaStreamTrackProcessor in a worker (prototype,
 *                          src/sensors/mstp-camera.ts; falls back to the <video> path)
 *
 * The mock* parameters only apply together with `mock`.
 */
import type { MockEngineOptions } from './mock-engine';

export interface BootParams {
  mock: boolean;
  /** MockEngine options (only with `mock`). */
  mockOptions: Pick<MockEngineOptions, 'speed' | 'failOn' | 'stallOn' | 'memoryLimitKeyframes'>;
  screen: 'diagnostics' | 'setup' | null;
  /** Camera frame path: 'video' (rVFC + createImageBitmap, default) or 'mstp'. */
  cameraPath: 'video' | 'mstp';
}

const FAIL_METHODS = ['init', 'frame', 'refine', 'densify', 'mesh', 'export', 'loadSession'] as const;
const STALL_METHODS = ['finishCapture', 'refine', 'densify', 'mesh'] as const;
type FailMethod = (typeof FAIL_METHODS)[number];
type StallMethod = (typeof STALL_METHODS)[number];

/** Text of a simulated failure in production builds (no URL-controlled text there). */
export const SIMULATED_FAILURE = 'Simulated failure';
const MAX_MESSAGE = 160;

/**
 * @param search `location.search`
 * @param dev    development build (`import.meta.env.DEV`): enables the custom `mockFail` message.
 */
export function parseBootParams(search: string, dev: boolean): BootParams {
  let q: URLSearchParams;
  try {
    q = new URLSearchParams(search);
  } catch {
    q = new URLSearchParams();
  }
  const mockParam = q.get('mock');
  const mock = mockParam === '1' || mockParam === 'true';
  const screenParam = q.get('screen');
  const screen = screenParam === 'diagnostics' || screenParam === 'setup' ? screenParam : null;
  const cameraPath = q.get('camera') === 'mstp' ? 'mstp' : 'video';
  const mockOptions: BootParams['mockOptions'] = {};
  if (mock) {
    const speed = Number(q.get('mockSpeed') ?? '1');
    mockOptions.speed = Number.isFinite(speed) && speed > 0 ? Math.min(50, Math.max(0.05, speed)) : 1;
    const fail = q.get('mockFail');
    if (fail) {
      const i = fail.indexOf(':');
      const method = i < 0 ? fail : fail.slice(0, i);
      if ((FAIL_METHODS as readonly string[]).includes(method)) {
        const custom = i < 0 ? '' : fail.slice(i + 1).replace(/[\u0000-\u001f\u007f]+/g, ' ').trim().slice(0, MAX_MESSAGE);
        mockOptions.failOn = { [method as FailMethod]: dev && custom ? custom : SIMULATED_FAILURE };
      }
    }
    const stall = (q.get('mockStall') ?? '').split(',').filter((m): m is StallMethod => (STALL_METHODS as readonly string[]).includes(m));
    if (stall.length) mockOptions.stallOn = Object.fromEntries(stall.map((m) => [m, true]));
    const limit = Number(q.get('mockMemoryLimit') ?? '');
    if (Number.isInteger(limit) && limit >= 1) mockOptions.memoryLimitKeyframes = limit;
  }
  return { mock, mockOptions, screen, cameraPath };
}
