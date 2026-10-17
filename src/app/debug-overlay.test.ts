import { describe, expect, it } from 'vitest';
import type { LiveDebug, LiveUpdate } from '../workers/protocol';
import { formatDebugLines } from './debug-overlay';

const debug: LiveDebug = {
  backend: 'webgpu',
  fallbackReason: null,
  featureInput: 'bitmap',
  srcWidth: 1440,
  srcHeight: 1920,
  width: 1440,
  height: 1920,
  numFeatures: 12,
  minFeatures: 150,
  sharpness: 3.2,
  sharpnessRef: 3.1,
  angularSpeed: 0.12,
  lumaMean: 0,
  lumaStd: 0,
  inliers: NaN,
  reason: 'only 12 features',
};

function update(d: Partial<LiveDebug> = {}): LiveUpdate {
  return { frameIndex: 7, decision: 'skip', guidance: 'more_texture', tracking: 'initializing', numKeyframes: 0, numRegistered: 0, numPoints: 0, timings: { total: 41.5, features: 20 }, debug: { ...debug, ...d } };
}

describe('formatDebugLines', () => {
  it('flags too few features and a blank frame', () => {
    const lines = formatDebugLines({ update: update(), camera: { width: 1920, height: 1440, frameRate: 30, label: 'Back Camera', frameMaxSide: null }, fps: 9.5, webgpuApi: true, logs: [] });
    expect(lines[0]).toBe('backend: WebGPU');
    expect(lines).toContain('camera: 1920×1440 @ 30 fps · Back Camera');
    expect(lines).toContain('!features: 12 / min 150 · input bitmap');
    expect(lines).toContain('!luma: 0 ± 0.0 (blank frame?)');
    expect(lines).toContain('reason: only 12 features');
  });

  it('shows the CPU fallback reason, a textured frame and engine warnings', () => {
    const lines = formatDebugLines({
      update: update({ backend: 'cpu', fallbackReason: 'self-test failed', numFeatures: 900, lumaMean: 120, lumaStd: 40 }),
      camera: null,
      fps: NaN,
      webgpuApi: false,
      logs: [{ type: 'log', level: 'warn', message: 'gray input' }],
    });
    expect(lines[0]).toBe('backend: CPU (self-test failed) · no navigator.gpu');
    expect(lines).toContain('features: 900 / min 150 · input bitmap');
    expect(lines).toContain('luma: 120 ± 40.0');
    expect(lines[lines.length - 1]).toBe('![warn] gray input');
  });

  it('waits for the first frame', () => {
    expect(formatDebugLines({ update: null, camera: null, fps: NaN, webgpuApi: true, logs: [] })).toEqual(['backend: –', 'waiting for the first frame…']);
  });
});
